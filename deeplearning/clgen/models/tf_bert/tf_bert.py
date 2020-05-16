# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import typing

from deeplearning.clgen.models.tf_bert import model
from deeplearning.clgen.models.tf_bert import optimizer
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen import telemetry
from deeplearning.clgen.tf import tf

from eupy.native import logger as l
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string(
    # "init_checkpoint", "/home/fivosts/test/model.ckpt-3000.index",
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

app.DEFINE_boolean("do_train", True, "Whether to run training.")

app.DEFINE_boolean("do_eval", True, "Whether to run eval on the dev set.")

app.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

app.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

app.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

app.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

app.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

app.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

app.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

app.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

app.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class tfBert(backends.BackendBase):

  def __init__(self, *args, **kwargs):

    super(tfBert, self).__init__(*args, **kwargs)
    self.bertConfig = {
          "vocab_size"                      : None,
          "hidden_size"                     : None,
          "num_hidden_layers"               : None,
          "num_attention_heads"             : None,
          "intermediate_size"               : None,
          "hidden_act"                      : None,
          "hidden_dropout_prob"             : None,
          "attention_probs_dropout_prob"    : None,
          "max_position_embeddings"         : None,
          "type_vocab_size"                 : None,
          "initializer_range"               : None,
    }

    self.max_seq_length                     = None
    self.train_batch_size                   = None
    self.eval_batch_size                    = None
    self.max_predictions_per_seq            = None
    self.learning_rate                      = None
    self.num_train_steps                    = None
    self.num_warmup_steps                   = None
    return

  def _ConfigModelParams(self):

    self.bertConfig = {
          "vocab_size"                      : self.atomizer.vocab_size,
          "hidden_size"                     : self.config.architecture.hidden_size,
          "num_hidden_layers"               : self.config.architecture.num_hidden_layers,
          "num_attention_heads"             : self.config.architecture.num_attention_heads,
          "intermediate_size"               : self.config.architecture.intermediate_size,
          "hidden_act"                      : self.config.architecture.hidden_act,
          "hidden_dropout_prob"             : 1.0 - self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob"    : 1.0 - self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"         : self.config.architecture.max_position_embeddings,
          "type_vocab_size"                 : self.config.architecture.type_vocab_size,
          "initializer_range"               : self.config.architecture.initializer_range,
    }

    self.max_seq_length                     = self.config.training.sequence_length
    self.train_batch_size                   = self.config.training.batch_size
    self.eval_batch_size                    = self.config.training.batch_size
    self.max_predictions_per_seq            = self.config.training.max_predictions_per_seq
    self.learning_rate                      = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_train_steps                    = self.config.training.num_train_steps
    self.num_warmup_steps                   = self.config.training.num_warmup_steps

    return

  def Train(self, corpus, **unused_kwargs) -> None:

    del unused_kwargs

    ## Initialize params and data generator
    self._ConfigModelParams()
    self.data_generator = data_generators.MaskLMBatchGenerator(
                                          corpus, self.config.training, self.cache.path, tf
                                          )
    self.num_steps_per_epoch = self.data_generator.num_batches
    self.num_epochs = int(self.num_train_steps / self.num_steps_per_epoch)
    self.ckpt_path = str(self.cache.path / "checkpoints")
    self.logfile_path = str(self.cache.path / "logs")

    self.telemetry = telemetry.TrainingLogger(self.logfile_path)

    ## Generate BERT Model from dict params
    bert_config = model.BertConfig.from_dict(self.bertConfig)

    l.getLogger().info("Checkpoint path: \n{}".format(self.ckpt_path))
    l.getLogger().info("Logging path: \n{}".format(self.logfile_path))
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = self.ckpt_path,
        save_checkpoints_steps = self.num_steps_per_epoch,
        save_summary_steps = self.num_steps_per_epoch,
        keep_checkpoint_max = 0,
        log_step_count_steps = self.num_steps_per_epoch,
        tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop = FLAGS.iterations_per_loop, ## TODO what is this ?
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host))

    model_fn = self._model_fn_builder(
        bert_config = bert_config,
        init_checkpoint = FLAGS.init_checkpoint, ## TODO Fix this to be assigned automatically
        learning_rate = self.learning_rate,
        num_train_steps = self.num_train_steps,
        num_warmup_steps = self.num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = self.train_batch_size,
        eval_batch_size = self.eval_batch_size)

    if FLAGS.do_train:
      l.getLogger().info("***** Running training *****")
      l.getLogger().info("  Batch size = {}".format(self.train_batch_size))
      train_input_fn = self.data_generator.generateTfDataset(
          max_seq_length = self.max_seq_length,
          num_cpu_threads = 8,
          is_training = True)

      l.getLogger().info("Running model for {} steps".format(self.num_train_steps))
      l.getLogger().info("Splitting {} steps into {} equivalent epochs, {} steps each".format(
                                        self.num_train_steps, self.num_epochs, self.num_steps_per_epoch
                                        )
                        )

      estimator.train(input_fn=train_input_fn, max_steps = self.num_train_steps)
      self.telemetry.TfRecordEpochs()

    if FLAGS.do_eval:
      l.getLogger().info("***** Running evaluation *****")
      l.getLogger().info("  Batch size = {}".format(self.eval_batch_size))

      eval_input_fn = self.data_generator.generateTfDataset(
          max_seq_length=self.max_seq_length,
          num_cpu_threads = 8,
          is_training=False)

      result = estimator.evaluate(
          input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

      output_eval_file = os.path.join(self.logfile_path, "eval_results.txt")
      with tf.io.gfile.GFile(output_eval_file, "w") as writer:
        l.getLogger().info("***** Eval results *****")
        for key in sorted(result.keys()):
          l.getLogger().info("  {} = {}".format(key, str(result[key])))
          writer.write("%s = %s\n" % (key, str(result[key])))


  def GetShortSummary(self) -> str:
    l.getLogger().debug("deeplearning.clgen.models.tf_sequential.tfSequential.GetShortSummary()")
    return (
      f"h_s: {self.bertConfig['hidden_size']}, "
      f"#h_l: {self.bertConfig['num_hidden_layers']}, "
      f"#att_h: {self.bertConfig['num_attention_heads']}, "
      f"imd_s: {self.bertConfig['intermediate_size']}, "
      f"h_act: {self.bertConfig['hidden_act']}, "
      f"{model_pb2.NetworkArchitecture.NeuronType.Name(self.config.architecture.backend)} "
      "network"
    )

  def _model_fn_builder(self,
                      bert_config, 
                      init_checkpoint, 
                      learning_rate,
                      num_train_steps,
                      num_warmup_steps,
                      use_tpu,
                      use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def _model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      # l.getLogger().info("*** Features ***")
      # for name in sorted(features.keys()):
      #   l.getLogger().info("  name = %s, shape = %s" % (name, features[name].shape))

      input_ids = features["input_ids"]
      # input_mask = features["input_mask"]
      # segment_ids = features["segment_ids"]
      masked_lm_positions = features["masked_lm_positions"]
      masked_lm_ids = features["masked_lm_ids"]
      masked_lm_weights = features["masked_lm_weights"]
      next_sentence_labels = features["next_sentence_labels"]

      is_training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)

      bert_model = model.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          # input_mask=input_mask, # You CAN ignore. Used for padding. 0s after real sequence. Now all 1s.
          # token_type_ids=segment_ids, # You can ignore. Used for double sentences (sA -> 0, sB ->1). Now all will be zero
          use_one_hot_embeddings=use_one_hot_embeddings)

      (masked_lm_loss,
       masked_lm_example_loss, masked_lm_log_probs) = self._get_masked_lm_output(
           bert_config, bert_model.get_sequence_output(), bert_model.get_embedding_table(),
           masked_lm_positions, masked_lm_ids, masked_lm_weights)

      (next_sentence_loss, next_sentence_example_loss,
       next_sentence_log_probs) = self._get_next_sentence_output(
           bert_config, bert_model.get_pooled_output(), next_sentence_labels)

      total_loss = masked_lm_loss + next_sentence_loss
      # print_op = tf.print(total_loss)
      # with tf.control_dependencies([print_op]):
      tvars = tf.compat.v1.trainable_variables()

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def _tpu_scaffold():
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = _tpu_scaffold
        else:
          l.getLogger().info("Loading model checkpoint from: {}".format(init_checkpoint))
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

      # l.getLogger().info("**** Trainable Variables ****")
      # for var in tvars:
      #   init_string = ""
      #   if var.name in initialized_variable_names:
      #     init_string = ", *INIT_FROM_CKPT*"
      #   l.getLogger().info("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))

      output_spec = None
      if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
        train_op = optimizer.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        training_hooks = self._GetSummaryHooks(save_steps = self.num_steps_per_epoch,
                                              output_dir = self.logfile_path,
                                              masked_lm_loss = masked_lm_loss,
                                              next_sentence_loss = next_sentence_loss,
                                              total_loss = total_loss,
                                              learning_rate = learning_rate
                                              )

        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode = mode,
            loss = total_loss,
            train_op = train_op,
            training_hooks = training_hooks,
            scaffold_fn = scaffold_fn)
      elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

        def _metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                      masked_lm_weights, next_sentence_example_loss,
                      next_sentence_log_probs, next_sentence_labels):
          """Computes the loss and accuracy of the model."""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions,
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.compat.v1.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          next_sentence_log_probs = tf.reshape(
              next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
          next_sentence_predictions = tf.argmax(
              next_sentence_log_probs, axis=-1, output_type=tf.int32)
          next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
          next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
              labels=next_sentence_labels, predictions=next_sentence_predictions)
          next_sentence_mean_loss = tf.compat.v1.metrics.mean(
              values=next_sentence_example_loss)

          return {
              "masked_lm_accuracy": masked_lm_accuracy,
              "masked_lm_loss": masked_lm_mean_loss,
              "next_sentence_accuracy": next_sentence_accuracy,
              "next_sentence_loss": next_sentence_mean_loss,
          }

        eval_metrics = (_metric_fn, [
            masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
            masked_lm_weights, next_sentence_example_loss,
            next_sentence_log_probs, next_sentence_labels
        ])
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            ## TODO check eval metrics!
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

      return output_spec

    return _model_fn


  def _get_masked_lm_output(self, 
                           bert_config,
                           input_tensor, 
                           output_weights, 
                           positions, 
                           label_ids,
                           label_weights
                           ):
    """Get loss and log probs for the masked LM."""
    input_tensor = self._gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
      # We apply one more non-linear transformation before the output layer.
      # This matrix is not used after pre-training.
      with tf.compat.v1.variable_scope("transform"):
        input_tensor = tf.compat.v1.layers.dense(
            input_tensor,
            units=bert_config.hidden_size,
            activation=model.get_activation(bert_config.hidden_act),
            kernel_initializer=model.create_initializer(
                bert_config.initializer_range))
        input_tensor = model.layer_norm(input_tensor)

      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = tf.compat.v1.get_variable(
          "output_bias",
          shape=[bert_config.vocab_size],
          initializer=tf.zeros_initializer())
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      label_ids = tf.reshape(label_ids, [-1])
      label_weights = tf.reshape(label_weights, [-1])

      one_hot_labels = tf.one_hot(
          label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
      numerator = tf.reduce_sum(label_weights * per_example_loss)
      denominator = tf.reduce_sum(label_weights) + 1e-5
      loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


  def _get_next_sentence_output(self, 
                               bert_config,
                               input_tensor,
                               labels
                              ):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("cls/seq_relationship"):
      output_weights = tf.compat.v1.get_variable(
          "output_weights",
          shape=[2, bert_config.hidden_size],
          initializer=model.create_initializer(bert_config.initializer_range))
      output_bias = tf.compat.v1.get_variable(
          "output_bias", shape=[2], initializer=tf.zeros_initializer())

      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      labels = tf.reshape(labels, [-1])
      one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      return (loss, per_example_loss, log_probs)


  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = model.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

  def _GetSummaryHooks(self, save_steps, output_dir, **kwargs):
    return [tf.estimator.SummarySaverHook(save_steps = save_steps,
                                          output_dir = output_dir,
                                          at_end = True,
                                          summary_op = [ tf.compat.v1.summary.scalar(name, value) 
                                                          for name, value in kwargs.items()
                                                        ]
                                          )
           ]
