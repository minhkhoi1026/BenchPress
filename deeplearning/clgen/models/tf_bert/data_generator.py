"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import sys
import time
import typing
import random
import progressbar
import collections
import copy
import glob
import humanize
import multiprocessing
import functools
import pickle

import numpy as np

from deeplearning.clgen.util import cache
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util.tf import tf
from deeplearning.clgen.proto import model_pb2
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "write_text_dataset", 
  False, 
  "Set True for MaskLM data generator to write dataset in text format, along with the tf_record."
)

flags.DEFINE_integer(
  "memory_limit",
  4,
  "Set maximum amount of available memory used for masking sequences in Gb. [Default]: 4",
)
flags.DEFINE_boolean(
  "force_remake_dataset",
  False,
  "Force data generator to re-mask encoded dataset and store tf_record."
)

class MaskSequence(typing.NamedTuple):
  """
  Tuple representation of a single MaskLM Instance. 
  This is not batch! generateTfDataset applies native batching,
  so this class represents a single instance!
  """

  seen_in_training     : np.int32
  original_input       : np.array
  input_ids            : np.array
  input_mask           : np.array
  masked_lm_positions  : np.array
  masked_lm_ids        : np.array
  masked_lm_weights    : np.array
  masked_lm_lengths    : np.array
  next_sentence_label  : np.int32

  @staticmethod
  def tfTypes():
    return (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int32)

  @staticmethod
  def npTypes():
    return (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.float32, np.int32, np.int32)

  @staticmethod
  def tfShapes(batch_size, sequence_length, max_position_embeddings = None):
    return (tf.TensorShape([batch_size, 1]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, 1]),
           )

  @staticmethod
  def estimatedSize(batch_size, sequence_length, max_position_embeddings = None):
    return (
      2 * np.zeros([batch_size, 1], dtype = np.int32).nbytes + 
      3 * np.zeros([batch_size, sequence_length]).nbytes + 
      4 * np.zeros([batch_size, max_position_embeddings]).nbytes
      )

  @property
  def sizeof_sequence(self):
    return (sys.getsizeof(self) + 
           self.seen_in_training.nbytes + self.original_input.nbytes + 
           self.input_ids.nbytes + self.input_mask.nbytes +
           self.masked_lm_positions.nbytes + self.masked_lm_ids.nbytes +
           self.masked_lm_weights.nbytes   + self.masked_lm_lengths.nbytes +
           self.next_sentence_label.nbytes
           )

  def shapeSeqToBatch(self, inp, batch_size):
    return "(" + str(batch_size) + ", " + ", ".join([str(s) for s in inp.shape]) + ")"

  def LogBatchTelemetry(self,
                        batch_size: int,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    l.getLogger().info("Step shape: seen_in_training: {}, original_input: {}, Input_ids: {}, input_mask: {}, masked_lm_positions: {}, masked_lm_ids: {}, masked_lm_weights: {}, masked_lm_lengths: {}, next_sentence_label: {}"
                        .format(self.shapeSeqToBatch(self.seen_in_training,     batch_size),
                                self.shapeSeqToBatch(self.original_input,       batch_size),
                                self.shapeSeqToBatch(self.input_ids,            batch_size),
                                self.shapeSeqToBatch(self.input_mask,           batch_size),
                                self.shapeSeqToBatch(self.masked_lm_positions,  batch_size),
                                self.shapeSeqToBatch(self.masked_lm_ids,        batch_size),
                                self.shapeSeqToBatch(self.masked_lm_weights,    batch_size),
                                self.shapeSeqToBatch(self.masked_lm_lengths,    batch_size),
                                self.shapeSeqToBatch(self.next_sentence_label,  batch_size),
                          )
                        )
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(self.sizeof_sequence * batch_size, binary = True),
              humanize.naturalsize(self.sizeof_sequence * batch_size * steps_per_epoch, binary = True),
              humanize.naturalsize(self.sizeof_sequence * batch_size * steps_per_epoch * num_epochs, binary = True),
          )
    )

def _holeSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_distribution: distributions.Distribution,
                  pickled_atomizer,
                  rngen: random.Random,
                  use_start_end: bool,
                  training_opts,
                  ) -> MaskSequence:
  """
  Inserts hole tokens to a given sequence.
  """
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  ## Tuple representation of mask id/position/hole_length for easy sorting
  class MaskedLmInstance():
    def __init__(self, 
                 pos_index: int, 
                 token_id: int, 
                 hole_length: int,
                 ):
      self.pos_index   = pos_index
      self.token_id    = token_id
      self.hole_length = hole_length

  # Unpack atomizer and sampler
  distribution = pickle.loads(pickled_distribution)
  atomizer     = pickle.loads(pickled_atomizer)

  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == atomizer.endToken)[0][0]
  elif padToken in seq:
    actual_length   = np.where(seq == atomizer.padToken)[0][0]
  else:
    actual_length   = len(seq)

  if use_start_end:
    candidate_indexes = np.arange(1, actual_length)
  else:
    candidate_indexes = np.arange(actual_length)
  rngen.shuffle(candidate_indexes)

  # total tokens to add in holes.
  # No more than max_predictions_per_seq (or otherwise specified), no less than actual seq length x the probability of hiding a token
  holes_to_predict  = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))

  # Flip input sequence to spread the hole lenghts to both directions uniformly.
  reverse_sequence = True if rngen.random() > 0.5 else False
  if reverse_sequence:
    input_ids         = list(np.copy(np.flip(seq)))
    candidate_indexes = (len(seq) - 1) - candidate_indexes
    actual_length     = len(seq) - (1 if use_start_end else 0)
  else:
    input_ids       = list(np.copy(seq))
  # List of (seq_idx, token_id, hole_length) tuples
  masked_lms        = []
  # Offset array. Indices represent elements in the initial array (seq)
  # Values of indices represent current offset position in processed array (input_ids).
  offset_idxs       = np.zeros(len(seq), dtype = np.int32)
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
  hole_length_list = []
  for pos_index in candidate_indexes:
    assert pos_index < len(seq), "Candidate index is out of bounds: {} >= {}".format(pos_index, len(seq))
    
    # Element in processed array can be found in its original index +/- offset
    input_id_idx = pos_index + offset_idxs[pos_index]
    if total_predictions >= holes_to_predict:
      break
    elif pos_index in visited_indices:
      # Do not target an index, already holed
      continue
    elif input_id_idx > len(seq):
      # Do not mask a part of input_ids that is going to be cropped.
      continue

    assert (input_ids[input_id_idx] == seq[pos_index], 
            "Original and offset-ted sequence have misaligned tokens: {}, {}"
            .format(seq[pos_index], input_ids[input_id_idx]))

    # Sampled number from distribution to represent the actual hole length
    hole_length = distribution.sample()
    # Inside range, make sure hole length does not run over input_id_idx bounds
    hole_length = min(hole_length, actual_length - input_id_idx)
    # Confirm there is no conflict with another hole, further down the sequence.
    for i in range(hole_length):
      if input_ids[input_id_idx + i] == atomizer.holeToken:
        hole_length = i
        break
    distribution.register(hole_length)
    hole_length_list.append(hole_length)
    
    # Target token for classifier is either the first token of the hole, or endholToken if hole is empty
    target = input_ids[input_id_idx] if hole_length > 0 else atomizer.endholeToken

    ## TODO. Think about '== self.atomizer.holeToken' condition.
    # if config.mask.random_placed_mask and hole_length != 0:
    #   if self.rngen.random() < 0.8:
    #     replacement_token = self.atomizer.holeToken
    #   else:
    #     if self.rngen.random() > 0.5:
    #       # Sometimes keep the original token.
    #       replacement_token = target
    #     else:
    #       # Other times add a random one.
    #       replacement_token = self.rngen.randint(0, self.atomizer.vocab_size - 1)
    # else:
    #   replacement_token = self.atomizer.holeToken
    replacement_token = atomizer.holeToken

    input_ids = (input_ids[:input_id_idx] + 
                 [replacement_token] + 
                 input_ids[input_id_idx + hole_length:])
    # This pos_index will get deprecated when someone before this index alters the offset array
    # So store position index, and after making all masks, update with updated offset array
    masked_lms.append(MaskedLmInstance(pos_index = pos_index, token_id = target, hole_length = hole_length))

    # Adjust the offset of all affected tokens, from pos_index and after.
    offset_idxs[pos_index + 1:] += 1 - hole_length
    # An empty hole is counted as a prediction of count 1.
    total_predictions           += max(1, hole_length)
    visited_indices.update(range(pos_index, pos_index + hole_length))

    for lm in masked_lms:
      ## TODO, remove this extensive-expensive check after you make sure that this function is bug free.
      test_index = lm.pos_index + offset_idxs[lm.pos_index]
      if input_ids[test_index] != atomizer.holeToken:
        assert False

  # Now update the entries with offset index.
  for lm in masked_lms:
    prev_index = lm.pos_index
    lm.pos_index = lm.pos_index + offset_idxs[lm.pos_index]

  # Un-reverse sequence
  if reverse_sequence:
    input_ids = list(reversed(input_ids))
    for lm in masked_lms:
      lm.pos_index = len(input_ids) - 1 - lm.pos_index

  # Now check that targets point only hole tokens
  for lm in masked_lms:
    assert input_ids[lm.pos_index] == atomizer.holeToken

  while len(input_ids) < len(seq):
    input_ids.append(atomizer.padToken)
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)
  masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []

  input_mask = np.ones(len(seq), dtype = np.int32)
  if atomizer.padToken in input_ids:
    first_pad_index = input_ids.index(atomizer.padToken)
    input_mask[first_pad_index:] = 0
    # Check that the pad index is likely correct.
    assert input_ids[first_pad_index] == atomizer.padToken, "{}".format(input_ids)
    assert input_ids[first_pad_index - 1] != atomizer.padToken

  seen_in_training    = np.int32(1 if train_set else 0)
  next_sentence_label = np.int32(0)
  """
    Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
    is intended on kernels. In any other case, check bert's create_instances_from_document
    to see how next_sentence_labels are calculated.
    Setting this to 0 means that next sentence is NOT random.
    Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.    
  """
  if len(masked_lms) == 0:
    l.getLogger().warn("No HOLE added to datapoint. Increase probability of hole occuring.")
  for p in masked_lms:
    if p.pos_index < len(seq):
      """
        Adding holes can increase or decrease the length of the original sequence.
        It is important in the end, to end up with an input sequence compatible
        with the model's sequence length, i.e. len(seq). If any mask is found 
        beyond that point, will have to be rejected.
      """
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
      masked_lm_weights.append(1.0)
      masked_lm_lengths.append(p.hole_length)
  num_holes = len(masked_lm_positions)
  while len(masked_lm_positions) < training_opts.max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(atomizer.padToken)
      masked_lm_weights.append(0.0)
      masked_lm_lengths.append(-1)

  assert (input_ids[:len(seq)].count(atomizer.holeToken) == num_holes,
    "Number of targets {} does not correspond to hole number in final input sequence: {}"
    .format(num_holes, input_ids[:len(seq)].count(atomizer.holeToken))
  )
  return MaskSequence(seen_in_training, seq,
                      np.asarray(input_ids[:len(seq)]), input_mask,
                      np.asarray(masked_lm_positions),  np.asarray(masked_lm_ids), 
                      np.asarray(masked_lm_weights),    np.asarray(masked_lm_lengths),
                      next_sentence_label 
                      ), hole_length_list

def _maskSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_atomizer,
                  rngen: random.Random,
                  training_opts,
                  config,
                  ) -> MaskSequence:
  """Inserts masks to a given sequence."""
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  ## Tuple representation of mask id/position for easy sorting
  class MaskedLmInstance(typing.NamedTuple):
    pos_index: int
    token_id: int

  # Unpack atomizer
  atomizer = pickle.loads(pickled_atomizer)

  # Actual length represents the sequence length before pad begins
  if atomizer.padToken in seq:
    actual_length = np.where(seq == atomizer.padToken)[0][0]
  else:
    actual_length = len(seq)

  candidate_indexes = np.arange(actual_length)
  rngen.shuffle(candidate_indexes)

  masks_to_predict = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))
  input_ids = list(np.copy(seq))
  masked_lms = []

  for pos_index in candidate_indexes:
    if len(masked_lms) >= masks_to_predict:
      break

    if config.mask.random_placed_mask:
      # 80% of the time, replace with [MASK]
      if rngen.random() < 0.8:
        input_ids[pos_index] = atomizer.maskToken
      else:
        # 10% of the time, keep original
        if rngen.random() < 0.5:
          pass
        # 10% of the time, replace with random word
        else:
          random_token = rngen.randint(0, atomizer.vocab_size - 1)
          while any(atomizer.vocab[t] == random_token for (idx, t) in atomizer.metaTokens.items()):
            random_token = rngen.randint(0, atomizer.vocab_size - 1)
          input_ids[pos_index] = rngen.randint(0, atomizer.vocab_size - 1)
    else:
      if rngen.random() < 0.8:
        input_ids[pos_index] = atomizer.maskToken

    masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

  assert len(masked_lms) <= masks_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []

  input_mask = np.ones(len(seq), dtype = np.int32)
  if atomizer.padToken in input_ids:
    input_mask[input_ids.index(atomizer.padToken):] = 0

  seen_in_training    = np.int32(1 if train_set else 0)
  next_sentence_label = np.int32(0)
  ## Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
  ## is intended on kernels. In any other case, check bert's create_instances_from_document
  ## to see how next_sentence_labels are calculated.
  ## Setting this to 0 means that next sentence is NOT random.
  ## Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.

  for p in masked_lms:
    masked_lm_positions.append(p.pos_index)
    masked_lm_ids.append(p.token_id)
    masked_lm_weights.append(1.0)
    masked_lm_lengths.append(1)
  while len(masked_lm_positions) < training_opts.max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(atomizer.padToken)
      masked_lm_weights.append(0.0)
      masked_lm_lengths.append(-1)

  return MaskSequence(seen_in_training, seq,
                      np.asarray(input_ids),           input_mask,
                      np.asarray(masked_lm_positions), np.asarray(masked_lm_ids), 
                      np.asarray(masked_lm_weights),   np.asarray(masked_lm_lengths),
                      next_sentence_label
                      )

class tfLMDataGenerator(object):
  def __init__(self):
    super(self, tfLMDataGenerator).__init__("tf_record")
    self.sampleBatch             = None # Extra
    self.sampleIndices           = None # Extra
    self.tfRecordSampler         = None # Extra
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    return super(self, tfLMDataGenerator).TrainMaskLMBatchGenerator(
              corpus, training_opts, cache_path
            )

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                sampler,
                                atomizer,
                                seed: int,
                                max_position_embeddings: int,
                                cache_path,
                                ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(self, tfLMDataGenerator).SampleMaskLMBatchGenerator(
          sampler, atomizer, seed, max_position_embeddings,
        )
    d.tfRecordSampler = d.tfRecordSampleGenerator() # Extra
    return d

  # Extra
  def generateTfDataset(self,
                      sequence_length: int,
                      is_training    : bool,
                      num_cpu_threads: int,
                      eval_set       : typing.List = None,
                      use_tpu        : bool = False,
                      ) -> "tf.Dataset":
    """Wrapper function that constructs a tf.Dataset used for training BERT."""

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for training.
      """
      def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        ## This function assumes record is still a file (expressed as TF dataset)
        ## It decodes this record to tf scalars.
        example = tf.io.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
          t = example[name]
          if t.dtype == tf.int64:
            t = tf.cast(t, dtype = tf.int32)
          example[name] = t
        return example

      batch_size = params["batch_size"]
      name_to_features = {
          "seen_in_training"        : tf.io.FixedLenFeature([1], tf.int64),
          "original_input"          : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "input_ids"               : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "input_mask"              : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "masked_lm_positions"     : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_ids"           : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_weights"       : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.float32),
          "masked_lm_lengths"       : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "next_sentence_labels"    : tf.io.FixedLenFeature([1], tf.int64),
      }

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        dataset = tf.io.gfile.glob([str(p) for p in self.dataset['train_dataset']['file']])
        d = tf.data.Dataset.from_tensor_slices(tf.constant(dataset))
        d = d.repeat()
        if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
          d = d.shuffle(buffer_size=len(dataset))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(dataset))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
          d = d.shuffle(buffer_size=100)
      else:
        if eval_set is None:
          dataset = tf.io.gfile.glob(
            [str(path) for tf_set in self.dataset for path in self.dataset[tf_set]['file']]
          )
        else:
          dataset = tf.io.gfile.glob([str(tf_set) for tf_set in eval_set])
        d = tf.data.TFRecordDataset(dataset)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

      # We must `drop_remainder` on training because the TPU requires fixed
      # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
      # and we *don't* want to drop the remainder, otherwise we wont cover
      # every sample.
      d = d.apply(
          tf.data.experimental.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              num_parallel_batches=num_cpu_threads,
              drop_remainder=use_tpu))
      return d
    return input_fn

  # Extra
  def generateTfSamples(self):
    """
    Contains input_fn closure function for estimator
    
    Returns:
      input_fn callable.
    """

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for inference.
      """

      def sample_gen(batch_size: int):
        """
        This generator yields iteratively the inference input blob for each step.
        In the first iteration, it yields sampler.encoded_start_text and then for each step,
        self.sampleBatch is updated with the model's current output through self.updateVocabulary.
        The generator stops when the model has filled all mask or hole tokens with predictions.

        Arguments:
          batch_size: The batch size used during inference.

        Yields:
          Current step's inference input for model.

        Returns:
          None
        """
        assert batch_size == len(self.sampleBatch), "{}, {}".format(batch_size, len(self.sampleBatch))
        original_input = [sample for sample in self.sampleBatch]
        while True:

          (input_ids, input_mask, masked_lm_positions,
          masked_lm_ids, masked_lm_weights, masked_lm_lengths) = [], [], [], [], [], []

          max_mask_len = max(
          [len(np.where(np.in1d(np.asarray(x), [self.atomizer.maskToken, self.atomizer.holeToken]))[0]) for x in self.sampleBatch]
          )
          if max_mask_len == 0:
            return
          for sample in self.sampleBatch:
            sample_masks = np.where(np.in1d(sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
            actual_mask_len = len(sample_masks)
            len_offset     = max_mask_len - actual_mask_len
            pad_idx      = np.where(sample == self.atomizer.padToken)[0]
            inp_mask     = np.ones(len(sample), dtype = np.int32)
            if len(pad_idx) > 0:
              inp_mask[pad_idx[0]:] = 0

            input_ids.append(list(sample))
            input_mask.append(list(inp_mask))
            masked_lm_positions.append(list(sample_masks) + [0] * len_offset)
            masked_lm_ids.append([self.atomizer.maskToken] * actual_mask_len + [self.atomizer.padToken] * len_offset)
            masked_lm_weights.append([0.0] * (actual_mask_len + len_offset))
            masked_lm_lengths.append([-1] * (actual_mask_len + len_offset))
          yield (np.full([batch_size, 1], -1), original_input, 
            input_ids, input_mask,
            masked_lm_positions, masked_lm_ids,
            masked_lm_weights, masked_lm_lengths,
            np.zeros([batch_size, 1]))

      batch_size = params['batch_size']
      sample = tf.data.Dataset.from_generator(
                lambda: sample_gen(batch_size), 
                output_types = MaskSequence.tfTypes(),
                output_shapes = MaskSequence.tfShapes(batch_size, self.sampler.sequence_length)
                )

      it = tf.compat.v1.data.make_one_shot_iterator(sample)
      (seen_in_training, original_input, 
        input_ids, input_mask,
        masked_lm_positions, masked_lm_ids,
        masked_lm_weights, masked_lm_lengths, next_sentence_labels) = it.get_next()

      return {
          'seen_in_training'      : seen_in_training,
          'original_input'        : original_input,
          'input_ids'             : input_ids,
          'input_mask'            : input_mask,
          'masked_lm_positions'   : masked_lm_positions,
          'masked_lm_ids'         : masked_lm_ids,
          'masked_lm_weights'     : masked_lm_weights,
          'masked_lm_lengths'     : masked_lm_lengths,
          'next_sentence_labels'  : next_sentence_labels,
      }
    return input_fn

  # Extra
  def tfRecordSampleGenerator(self):

    if self.sampler.isFixedStr:
      return None

    assert not self.sampler.config.HasField("start_text")

    if self.sampler.config.HasField("train_set"):
      sampledDataset = "train_dataset"
    elif self.sampler.config.HasField("validation_set"):
      sampledDataset = "validation_dataset"
    elif self.sampler.config.HasField("sample_set"):
      sampledDataset = "pred_{}_{}".format(
        self.sampler.config.sample_set.max_predictions_per_seq,
        "mask" if self.sampler.config.sample_set.HasField("mask") 
               else "hole_{}".format(self.sampler.config.sample_set.hole.hole_length)
      )
    path_list = glob.glob(str(self.cache.path / "{}_*.{}".format(sampledDataset, self.file_extension)))
    if len(path_list) == 0:
      raise FileNotFoundError(path_list)

    for path in path_list:
      for example in tf.compat.v1.io.tf_record_iterator(path):
        input_ids = np.asarray(tf.train.Example.FromString(example).features.feature['input_ids'].int64_list.value)
        if self.atomizer.padToken in input_ids:
          yield input_ids[:np.where(input_ids == self.atomizer.padToken)[0][0]]
        else:
          yield input_ids

  def InitSampleBatch(self) -> None:
    """
      Initializes data_generator for inference.
      self.sampleBatch is initialized with sampler.encoded_start_text
    """
    if not self.sampler.isFixedStr:
      try:
        start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
      except StopIteration:
        l.getLogger().warn("Repeating iterator on dataset...")
        self.tfRecordSampler = self.tfRecordSampleGenerator()
        try:
          start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
        except Exception as e:
          raise e
      self.sampler.setStartText(self.atomizer.DeatomizeIndices(start_text))
      self.sampler.Specialize(self.atomizer)
    
    assert self.sampler.sequence_length <= self.max_position_embeddings, "Sampler sequence length exceeds max position embeddings."
    input_sample = self.sampler.encoded_start_text
    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

    target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
    assert len(target_idx) != 0, "No target prediction in sample text"

    num_masks = np.count_nonzero(input_sample == self.atomizer.maskToken)
    num_holes = np.count_nonzero(input_sample == self.atomizer.holeToken)
    num_targets = num_masks + num_holes

    padded_sample = self._padToMaxPosition(input_sample)
    padded_sample = padded_sample[:self.sampler.sequence_length]
    self.sampleBatch   = np.repeat(padded_sample[None, :], self.sampler.batch_size, axis = 0)
    self.sampleIndices = [[[] for i in range(num_targets)] for j in range(self.sampler.batch_size)]
    return

  def updateSampleBatch(self, 
                        input_ids     : np.array,
                        masked_lm_ids : np.array,
                        ) -> np.array:
    """
    Updates self.sampleBatch with the model's output prediction.
    The output, if still contains hole or mask tokens, is fed back
    to the model's input through the input_fn's sample_gen generator.
    """
    assert len(input_ids) == len(masked_lm_ids), "Inputs and predictions do not have the same batch size."

    updated_sequence = []
    done = True
    for batch_idx, _ in enumerate(input_ids):
      batch = []
      mask_id_index     = 0
      closed_hole_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.atomizer.maskToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
          mask_id_index += 1
          batch.append(mt)
        elif token == self.atomizer.holeToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
          mask_id_index += 1
          if mt != self.atomizer.endholeToken:
            batch.append(mt)
            batch.append(self.atomizer.holeToken)
            done = False
        else:
          batch.append(token)
      batch = np.asarray(batch)
      batch = self._padToMaxPosition(batch)
      # TODO, chop sequence for now, but TODO it: 
      # If a sequence is bigger than it should, crop one or both edges,
      # save them and send max_position_embeddings for next step.
      # Then, concat it back.
      if self.sampler.sequence_length > len(batch):
        l.getLogger().warn("Cropped {} tokens from sample batch".format(self.sampler.sequence_length - len(batch)))
      batch = batch[:self.sampler.sequence_length]
      updated_sequence.append(batch)

    self.sampleBatch = np.asarray(updated_sequence)
    return self.sampleBatch, self.sampleIndices

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""
     
    writer = tf.io.TFRecordWriter(str(masked_corpus['file']))
    if FLAGS.write_text_dataset:
      file_writer = open(masked_corpus['txt'], 'w')

    for (inst_index, instance) in enumerate(masked_corpus['corpus']):
      seen_in_training = instance.seen_in_training
      original_input   = instance.original_input
      input_ids        = instance.input_ids
      input_mask       = instance.input_mask

      assert len(input_ids) == self.training_opts.sequence_length, "len(input_ids):  {}, sequence_length: {}".format(len(input_ids), self.training_opts.sequence_length)

      masked_lm_positions   = instance.masked_lm_positions
      masked_lm_ids         = instance.masked_lm_ids
      masked_lm_weights     = instance.masked_lm_weights
      masked_lm_lengths     = instance.masked_lm_lengths
      next_sentence_label   = instance.next_sentence_label
      features              = collections.OrderedDict()

      features["seen_in_training"]      = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list([seen_in_training])))

      features["original_input"]        = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(original_input)))

      features["input_ids"]             = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_ids)))

      features["input_mask"]            = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_mask)))

      features["masked_lm_positions"]   = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_positions)))

      features["masked_lm_ids"]         = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_ids)))

      features["masked_lm_weights"]     = tf.train.Feature(float_list = tf.train.FloatList(
                                                                value = list(masked_lm_weights)))

      features["masked_lm_lengths"]     = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_lengths)))

      features["next_sentence_labels"]  = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list([next_sentence_label])))

      tf_example = tf.train.Example(features = tf.train.Features(feature = features))
      writer.write(tf_example.SerializeToString())
      if FLAGS.write_text_dataset:
        file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'masked_lm_positions': {}\n'masked_lm_ids': {}\n'masked_lm_weights': {}\n'masked_lm_lengths': {}\n'next_sentence_labels': {}\n\n"
                            .format((True if seen_in_training == 1 else False),
                                    self.atomizer.DeatomizeIndices(original_input, ignore_token = self.atomizer.padToken),
                                    self.atomizer.DeatomizeIndices(input_ids, ignore_token = self.atomizer.padToken),
                                    input_mask, 
                                    masked_lm_positions, 
                                    self.atomizer.DeatomizeIndices(masked_lm_ids), 
                                    masked_lm_weights, 
                                    masked_lm_lengths, 
                                    next_sentence_label)
                            )
    writer.close()
    if FLAGS.write_text_dataset:
      file_writer.close()
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                      .format(inst_index + 1, self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return
