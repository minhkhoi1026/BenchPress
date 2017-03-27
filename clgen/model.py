#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
CLgen model.
"""
import numpy as np
import os
import progressbar
import re
import sys
import tarfile
import time

from copy import deepcopy
from glob import iglob
from labm8 import fs
from labm8 import system
from labm8 import time as labtime
from six import string_types
from six.moves import cPickle
from tempfile import mktemp

import clgen
from clgen import cache
from clgen import config as cfg
from clgen import lockfile
from clgen import log
from clgen.cache import Cache
from clgen.corpus import Corpus


def get_default_author() -> str:
    """
    Get the default author name for CLgen dist models.

    If CLGEN_AUTHOR environment variable is set, use that. Else, author
    is $USER@$HOSTNAME.

    Returns:
        str: Author name.
    """
    return os.environ.get(
        "CLGEN_AUTHOR",
        "{user}@{host}".format(user=system.USERNAME, host=system.HOSTNAME))


# Default options used for model. Any values provided by the user will override
# these defaults.
DEFAULT_MODEL_OPTS = {
    "author": get_default_author(),
    "architecture": {
      "model_type": "lstm",  # {lstm,rnn.gru}
      "rnn_size": 128,  # num nodes in layer
      "num_layers": 2,  # num layers
    },
    "train_opts": {
      "epochs": 10,
      "grad_clip": 5,
      "learning_rate": 2e-3,  # initial learning rate
      "lr_decary_rate": 5,  # % to reduce learning rate by per epoch
      "intermediate_checkpoints": True
    }
}


class ModelError(clgen.CLgenError):
    """
    Module level error
    """
    pass


def from_json(model_json: dict):
    """
    Load model from JSON.

    Arguments:
        model_json (dict): JSON specification.

    Returns:
        Model: Model instance.
    """
    assert(type(model_json) is dict)

    if "corpus" not in model_json:
        raise clgen.UserError("model JSON has no corpus entry")

    # create corpus and remove from JSON
    corpus = Corpus.from_json(model_json.pop("corpus"))

    return Model(corpus, **model_json)


class Model(clgen.CLgenObject):
    """
    A CLgen Model.
    """
    def __init__(self, corpus: Corpus, **opts):
        """
        Instantiate model.

        Arguments:
            corpus (Corpus): Corpus instance.
            opts (dict): Training options.
        """
        assert(isinstance(corpus, Corpus))

        # Validate options
        for key in opts.keys():
            if key not in DEFAULT_MODEL_OPTS:
                raise clgen.UserError(
                    "Unsupported model option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_MODEL_OPTS.keys()))))

        # set properties
        self.opts = clgen.update(deepcopy(DEFAULT_MODEL_OPTS), opts)
        self.corpus = corpus
        self.hash = self._hash(self.corpus, self.opts)
        self.cache = Cache(fs.path("model", self.hash))

        log.debug("model", self.hash)

        # validate metadata against cache
        meta = self.to_json()
        if self.cache["META"]:
            cached_meta = clgen.load_json_file(self.cache["META"])
            if meta != cached_meta:
                raise clgen.InternalError("model metadata mismatch")
        else:
            clgen.write_json_file(self.cache.keypath("META"), meta)

    def _hash(self, corpus: Corpus, opts: dict) -> str:
        """ compute model hash """
        hashopts = deepcopy(opts)
        hashopts["train_opts"].pop("epochs")
        return clgen.checksum_list(corpus.hash, *clgen.dict_values(hashopts))

    def _init_tensorflow(self, infer: bool=False):
        """
        Deferred importing of tensorflow and initializing model for training
        or sampling.

        This is necessary for two reasons: first, the tensorflow graph is
        different for training and inference, so must be reset when switching
        between modes. Second, importing tensorflow takes a long time, so
        we only want to do it if we actually need to.

        Arguments:
            infer (bool): If True, initialize model for inference. If False,
                initialize model for training.

        Returns:
            module: imported TensorFlow module
        """
        if cfg.USE_CUDA:
            import setGPU

        import tensorflow as tf
        import tensorflow.contrib.legacy_seq2seq as seq2seq
        from tensorflow.contrib import rnn

        # Use self.tensorflow_state to mark whether or not model is configured
        # for training or inference.
        try:
            if self.tensorflow_state == infer:
                return tf
        except AttributeError:
            pass

        self.cell_fn = {
            "lstm": rnn.BasicLSTMCell,
            "gru": rnn.GRUCell,
            "rnn": rnn.BasicRNNCell
        }.get(self.model_type, None)
        if self.cell_fn is None:
            raise clgen.UserError("Unrecognized model type")

        # reset the graph when switching between training and inference
        tf.reset_default_graph()

        # corpus info:
        batch_size = 1 if infer else self.corpus.batch_size
        seq_length = 1 if infer else self.corpus.seq_length
        vocab_size = self.corpus.vocab_size

        fs.mkdir(self.cache.path)

        cell = self.cell_fn(self.rnn_size, state_is_tuple=True)
        self.cell = cell = rnn.MultiRNNCell([cell] * self.num_layers,
                                                 state_is_tuple=True)
        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        scope_name = 'rnnlm'
        with tf.variable_scope(scope_name):
            softmax_w = tf.get_variable("softmax_w",
                                        [self.rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding",
                                            [vocab_size, self.rnn_size])
                inputs = tf.split(
                    axis=1, num_or_size_splits=seq_length,
                    value=tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(
            inputs, self.initial_state, cell,
            loop_function=loop if infer else None, scope=scope_name)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * seq_length])],
            vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # set model status
        self.tensorflow_state = infer

        return tf


    def _get_params_path(self, ckpt) -> str:
        """ return path to checkpoint closest to target num of epochs """
        paths = ckpt.all_model_checkpoint_paths
        batch_nums = [int(x.split('-')[-1]) for x in paths]
        epoch_nums = [int((x + 1) / (self.corpus.num_batches))
                      for x in batch_nums]

        closest = self.epochs
        closest_path = None
        for e, path in zip(epoch_nums, paths):
            diff = self.epochs - e
            if diff >= 0 and diff < closest:
                log.verbose("  cached checkpoint at epoch =", e, "diff =", diff)
                closest = diff
                closest_path = path

        return closest_path, paths


    def _locked_train(self, quiet):
        tf = self._init_tensorflow(infer=False)

        # training options
        learning_rate = self.train_opts["learning_rate"]
        decay_rate = self.train_opts["lr_decary_rate"]
        checkpoint_path = fs.path(self.cache.path, "model.ckpt")

        # resume from prior checkpoint
        ckpt_path, ckpt_paths = None, None
        if self.checkpoint_path:
            # check if all necessary files exist
            assert(fs.isdir(self.checkpoint_path))
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            assert(ckpt)
            assert(ckpt.model_checkpoint_path)
            ckpt_path, ckpt_paths = self._get_params_path(ckpt)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # keep all checkpoints
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # restore model from closest checkpoint
            if ckpt_path:
                log.debug("restoring", ckpt_path)
                saver.restore(sess, ckpt_path)
                log.info("restored checkpoint {}".format(ckpt_path))

            # make sure we don't lose track of other checkpoints
            if ckpt_paths:
                saver.recover_last_checkpoints(ckpt_paths)

            start_batch = sess.run(self.epoch) * self.corpus.num_batches
            max_batch = self.epochs * self.corpus.num_batches

            # progress bar
            bar = progressbar.ProgressBar(max_value=max_batch)

            if sess.run(self.epoch) != self.epochs:
                log.info("training", self)

            for e in range(sess.run(self.epoch) + 1, self.epochs + 1):

                # decay and set learning rate
                new_learning_rate = learning_rate * (
                    (float(100 - decay_rate) / 100.0) ** (e - 1))
                sess.run(tf.assign(self.learning_rate, new_learning_rate))
                sess.run(tf.assign(self.epoch, e))

                self.corpus.create_batches()

                state = sess.run(self.initial_state)
                for b in range(self.corpus.num_batches):
                    x, y = self.corpus.next_batch()
                    feed = {self.input_data: x, self.targets: y}
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    train_loss, state, _ = sess.run(
                        [self.cost, self.final_state, self.train_op], feed)

                    # update progress bar
                    batch_num = (e - 1) * self.corpus.num_batches + b
                    bar.update(batch_num)

                save = self.opts["train_opts"]["intermediate_checkpoints"]
                save |= e == self.epochs  # always save on last epoch
                if save:
                    saver.save(sess, checkpoint_path, global_step=batch_num)

                    next_checkpoint = e * self.corpus.num_batches + b
                    max_epoch = self.epochs
                    log.info("\n{self} epoch {e} / {max_epoch}. "
                             "next checkpoint at batch {next_checkpoint}"
                             .format(**vars()))

        return self

    def train(self, quiet: bool=False):
        """
        Train model.

        Arguments:
            quiet (bool, optional): If true, print less.

        Returns:
            Model: self.
        """
        with self.lock.acquire():
            return self._locked_train(quiet)


    def sample(self, seed_text: str="__kernel void", output=sys.stdout,
               num_samples: int=1, temperature: float=1, max_length: int=10000,
               seed: int=None, quiet: bool=False) -> None:
        """
        Sample model.

        Arguments:
            seed_text (str, optional): Sample start text
            output (file handler, optional): Where to print output to
            num_samples (int, optional): Number of samples to generated
            temperature (float, optional): Sampling temperature
            max_length (int, optional): Maximum sample length
            seed (int, optional): If set, set random number seed for
                reproducible samples. If None, set no seed.
            quiet (bool, optional): If False, stream output to stdout
        """
        if self.lock.islocked:
            raise lockfile.UnableToAcquireLockError(self.lock)

        tf = self._init_tensorflow(infer=True)

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.cache.path)

            assert(ckpt)
            assert(ckpt.model_checkpoint_path)

            saver.restore(sess, ckpt.model_checkpoint_path)

            def weighted_pick(weights, temperature):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return int(np.searchsorted(t, np.random.rand(1) * s))

            start_depth = 0
            start_started = False
            for char in seed_text:
                if char == '{':
                    start_depth += 1
                    start_started = True
                elif char == '}':
                    start_depth -= 1

            for i in range(1, num_samples + 1):
                state = sess.run(self.cell.zero_state(1, tf.float32))
                depth = start_depth  # function block depth
                started = start_started

                seed_tensor = self.corpus.atomizer.atomize(seed_text)
                for index in seed_tensor[:-1]:
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {self.input_data: x, self.initial_state: state}
                    [state] = sess.run([self.final_state], feed)

                sampling_type = 1  # default
                output.write("\n\n/* SAMPLE {} */\n\n".format(i))
                output.write(seed_text)
                if not quiet:
                    sys.stdout.write("\n\n/* SAMPLE {} */\n\n".format(i))
                    sys.stdout.write(seed_text)
                    sys.stdout.flush()

                index = seed_tensor[-1]

                for _ in range(max_length):
                    x = np.zeros((1, 1))
                    x[0, 0] = index
                    feed = {self.input_data: x, self.initial_state: state}
                    [probs, state] = sess.run(
                        [self.probs, self.final_state], feed)
                    p = probs[0]

                    # sample distribution to pick next:
                    index = weighted_pick(p, temperature)
                    # alternatively, select most probable:
                    # index = np.argmax(p)

                    atom = self.corpus.atomizer.deatomize([index])
                    output.write(atom)
                    if not quiet:
                        sys.stdout.write(atom)

                    # update function block depth
                    for char in atom:
                        if char == '{':
                            started = True
                            depth += 1
                        elif char == '}':
                            depth -= 1
                    # stop sampling if depth = 0
                    if started and depth < 1:
                        break

            if not quiet:
                sys.stdout.write('\n\n')

    @property
    def lock(self):
        lockpath = self.cache.keypath("LOCK")
        return lockfile.LockFile(lockpath)

    @property
    def model_type(self) -> str:
        return self.opts["architecture"]["model_type"]

    @property
    def rnn_size(self) -> int:
        return self.opts["architecture"]["rnn_size"]

    @property
    def num_layers(self) -> int:
        return self.opts["architecture"]["num_layers"]

    @property
    def grad_clip(self) -> int:
        return self.train_opts["grad_clip"]

    @property
    def epochs(self) -> int:
        return self.train_opts["epochs"]

    @property
    def train_opts(self) -> dict:
        return self.opts["train_opts"]

    @property
    def meta(self) -> dict:
        """
        Get trained model metadata.

        Format spec: https://github.com/ChrisCummins/clgen/issues/25

        Returns:
            dict: Metadata.
        """
        # checksum corpus and model cache files. Paths are relative to cache
        # root.
        cache_root_re = r'^' + cache.ROOT + '/'
        corpus_files = dict(
            (re.sub(cache_root_re, "", x), clgen.checksum_file(x))
            for x in fs.ls(self.corpus.cache.path, abspaths=True))
        model_files = dict(
            (re.sub(cache_root_re, "", x), clgen.checksum_file(x))
            for x in fs.ls(self.cache.path, abspaths=True))

        contents = corpus_files.copy()
        contents.update(model_files)

        _meta = deepcopy(self.opts)
        _meta["version"] = clgen.version()
        _meta["date_packaged"] = labtime.nowstr()
        _meta["corpus"] = self.corpus.meta,
        _meta["contents"] = contents

        return _meta

    def __repr__(self) -> str:
        """
        String representation.
        """
        hash = self.hash
        size = self.rnn_size
        nlayers = self.num_layers
        model = self.model_type.upper()
        nepochs = self.epochs

        return "model[{hash}]: {size}x{nlayers}x{nepochs} {model}".format(**vars())

    def to_json(self) -> dict:
        d = deepcopy(self.opts)
        d["corpus"] = self.corpus.to_json()
        return d

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Model):
            return False
        return rhs.hash == self.hash

    def __ne__(self, rhs) -> bool:
        return not self.__eq__(rhs)

    @property
    def checkpoint_path(self) -> str:
        """
        Get path to most checkpoint, if exists.

        Returns:

            str or None: Path to checkpoint, or None if no checkpoints.
        """
        if self.cache["checkpoint"]:
            return self.cache.path
        else:
            return None

    @staticmethod
    def from_json(model_json: dict):
        return from_json(model_json)
