"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib
import functools
import typing
import tqdm
import multiprocessing
import numpy as np

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import grewe
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import logging as l

def ExtractorWorker(cldrive_entry: cldrive.CLDriveSample, fspace: str):
  """
  Worker that extracts features and buffers cldrive entry, to maintain consistency
  among multiprocessed data.
  """
  features = extractor.ExtractFeatures(cldrive_entry.source, [fspace])
  if fspace in features and features[fspace]:
    return features[fspace], cldrive_entry
  return None

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str, corpus_path: pathlib.Path, random_seed: int) -> "DownstreamTask":
    return TASKS[task](corpus_path, random_seed)

  def __init__(self, name: str, random_seed: int) -> None:
    self.name        = name
    self.random_seed = random_seed
    return

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  This class is responsible to fetch the raw data and act as a tokenizer
  for the data. Reason is, the data generator should be agnostic of the labels.
  """
  @property
  def input_size(self) -> int:
    return 4
  
  @property
  def input_labels(self) -> typing.List[str]:
    return [
      "tr_bytes/(comp+mem)",
      "coalesced/mem",
      "localmem/(mem+wgsize)",
      "comp/mem"
    ]

  @property
  def output_size(self) -> int:
    return 2

  @property
  def output_labels(self) -> typing.Tuple[str, str]:
    return ["CPU", "GPU"]

  @property
  def feature_space(self) -> str:
    return "GreweFeatures"

  def __init__(self, corpus_path: pathlib.Path, random_seed: int) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive", random_seed)
    self.corpus_path    = corpus_path
    self.setup_dataset()
    self.data_generator = data_generator.ListTrainDataloader(self.dataset)

    ## Setup random seed np random stuff
    self.seed_generator = np.random
    self.seed_generator.seed(random_seed)
    self.rand_generators = {}
    max_fval = {
      'comp'      : 300,
      'rational'  : 50,
      'mem'       : 50,
      'localmem'  : 50,
      'coalesced' : 10,
      'atomic'    : 10,
    }
    for fk in grewe.KEYS:
      if fk not in {"F2:coalesced/mem", "F4:comp/mem"}:
        seed = self.seed_generator.randint(0, 2**32-1)
        rgen = np.random
        rgen.seed(seed)
        low_bound = 1 if fk in {"comp", "mem"} else 0
        up_bound  = max_fval[fk]
        self.rand_generators[fk] = lambda: rgen.randint(low_bound, up_bound)
    return

  def __repr__(self) -> str:
    return "GrewePredictive"

  def setup_dataset(self) -> None:
    """
    Fetch data and preprocess into corpus for Grewe's predictive model.
    """
    self.dataset = []
    self.corpus_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus_path)), must_exist = True)
    data    = [x for x in self.corpus_db.get_valid_data(dataset = "GitHub")]
    pool = multiprocessing.Pool()
    it = pool.imap_unordered(functools.partial(ExtractorWorker, fspace = "GreweFeatures"), data)
    idx = 0
    try:
      for dp in tqdm.tqdm(it, total = len(data), desc = "Grewe corpus setup", leave = False):
        if dp:
          feats, entry = dp
          self.dataset.append(
            (
              self.InputtoEncodedVector(feats, entry.transferred_bytes, entry.local_size),
              [self.TargetLabeltoID(entry.status)]
            )
          )
          idx += 1
        if idx >= 100:
          break
      pool.close()
    except Exception as e:
      pool.terminate()
      raise e
    pool.terminate()
    return

  def CollectRuntimeFeatures(self,
                             samples: typing.List['ActiveSample'],
                             top_k  : int
                            ) -> typing.List['ActiveSample']:
    """
    Collect the top_k samples that can run on CLDrive and set their global size
    to the appropriate value so it can match the transferred bytes.
    """
    new_samples = []
    total = 0
    for sample in sorted(samples, key = lambda x: x.score):
      exp_tr_bytes = sample.runtime_features['transferred_bytes']
      local_size   = sample.runtime_features['local_size']
      found        = False
      gsize        = 1
      prev         = math.inf
      while not found and gsize <= 20:
        sha256 = crypto.sha256_str(sample.text + "BenchPress" + str(2**gsize) + str(local_size))
        if sha256 in self.corpus_db.status_cache:
          cached = self.corpus_db.get_entry(sample.text, "BenchPress", 2**gsize, local_size)
        else:
          cached = self.corpus_db.update_and_get(
            sample.text,
            "BenchPress",
            global_size = 2**gsize,
            local_size  = local_size,
            num_runs    = 10,
            timeout     = 15,
          )
        if cached.status in {"CPU", "GPU"}:
          tr_bytes = cached.transferred_bytes
        else:
          tr_bytes = None
        if tr_bytes:
          if tr_bytes < exp_tr_bytes:
            gsize += 1
            prev = tr_bytes
          else:
            found = True
            if abs(exp_tr_bytes - tr_bytes) > abs(exp_tr_bytes - prev):
              gsize -= 1
              tr_bytes  = abs(exp_tr_bytes - prev)
            else:
              tr_bytes  = abs(exp_tr_bytes - tr_bytes)
        else:
          gsize += 1
      if found:
        new_runtime_feats = sample.runtime_features
        new_runtime_feats['transferred_bytes'] = tr_bytes
        new_runtime_feats['global_size'] = 2**gsize
        cached = self.corpus_db.update_and_get(
          sample.text,
          "BenchPress",
          global_size = new_runtime_feats['global_size'],
          local_size  = new_runtime_feats['local_size'],
          num_runs    = 1000,
          timeout     = 60,
        )
        new_runtime_feats['label'] = cached.label
        new_samples.append(sample._replace(runtime_features = new_runtime_feats))
        total += 1
      if top_k != -1 and total >= top_k:
        break
    return new_samples

  def UpdateDataGenerator(self,
                          new_samples: typing.List['ActiveSample'],
                          top_k: int,
                          ) -> data_generator.ListTrainDataloader:
    """
    Collect new generated samples, find their runtime features and processs to a torch dataset.
    """
    new_samples = self.CollectRuntimeFeatures(new_samples, top_k)
    updated_dataset = [
      (
        self.InputtoEncodedVector(feats,
                                  entry.runtime_features['transferred_bytes'],
                                  entry.runtime_features['local_size']
                                  ),
        [self.TargetLabeltoID(entry.runtime_features['label'])]
      ) for entry in new_samples
    ]
    ## Add new samples along with 100 old random samples.
    extra_steps = 100
    keys = set(np.RandomState().randint(0, len(self.dataset)) for _ in range(extra_steps))
    updated_dataset += [x for idx, x in self.dataset if idx in keys]
    l.logger().warn("What happens if all runtime features crash and you don't have any new samples?")
    return new_samples, data_generator.ListTrainDataloader(updated_dataset)

  def sample_space(self, num_samples: int = 512) -> data_generator.DictPredictionDataloader:
    """
    Go fetch Grewe Predictive model's feature space and randomly return num_samples samples
    to evaluate. The predictive model samples are mapped as a value to the static features
    as a key.
    """
    l.logger().warn("Assuming wgsize (local size) and transferred_bytes is very problematic.")
    samples = []
    for x in range(num_samples):
      fvec = {
        k: self.rand_generators[k]()
        for k in grewe.KEYS if k not in {"F2:coalesced/mem", "F4:comp/mem"}
      }
      try:
        fvec['F2:coalesced/mem'] = fvec['coalesced'] / fvec['mem']
      except ZeroDivisionError:
        fvec['F2:coalesced/mem'] = 0.0
      try:
        fvec['F4:comp/mem'] = fvec['comp'] / fvec['mem']      
      except ZeroDivisionError:
        fvec['F4:comp/mem'] = 0.0
      samples.append(
        {
          'static_features'  : self.StaticFeatDictToVec(fvec),
          'runtime_features' : [80000, 256],
          'input_ids'        : self.InputtoEncodedVector(fvec, 80000, 256)
        }
      )
    return data_generator.DictPredictionDataloader(samples)

  def StaticFeatDictToVec(self, static_feats: typing.Dict[str, float]) -> typing.List[float]:
    """
    Process grewe static features dictionary into list of floats to be passed as tensor.
    """
    return [static_feats[key] for key in grewe.KEYS]

  def VecToStaticFeatDict(self, feature_values: typing.List[float]) -> typing.Dict[str, float]:
    """
    Process float vector of feature values to dictionary of features.
    """
    return {key: val for key, val in zip(grewe.KEYS, feature_values)}

  def VecToRuntimeFeatDict(self, runtime_values: typing.List[int]) -> typing.Dict[str, int]:
    """
    Process runtime int values to runtime features dictionary.
    """
    trb, ls = runtime_values
    return {
      'transferred_bytes' : trb,
      'local_size'        : ls,
    }

  def InputtoEncodedVector(self,
                           static_feats      : typing.Dict[str, float],
                           transferred_bytes : int,
                           local_size       : int,
                           ) -> typing.List[float]:
    """
    Encode consistently raw features to Grewe's predictive model inputs.
    """
    try:
      i1 = transferred_bytes / (static_feats['comp'] + static_feats['mem'])
    except ZeroDivisionError:
      i1 = 0.0
    try:
      i2 = static_feats['coalesced'] / static_feats['mem']
    except ZeroDivisionError:
      i2 = 0.0
    try:
      i3 = static_feats['localmem'] / (static_feats['mem'] * local_size)
    except ZeroDivisionError:
      i3 = 0.0
    try:
      i4 = static_feats['comp'] / static_feats['mem']
    except ZeroDivisionError:
      i4 = 0.0
    return [i1, i2, i3, i4]

  def TargetIDtoLabels(self, id: int) -> str:
    """
    Integer ID to label of predictive model.
    """
    return {
      0: "CPU",
      1: "GPU",
    }[id]

  def TargetLabeltoID(self, label: str) -> int:
    """
    Predictive label to ID.
    """
    return {
      "CPU": 0,
      "GPU": 1,
    }[label]

  def TargetLabeltoEncodedVector(self, label: str) -> typing.List[int]:
    """
    Label to target vector.
    """
    return {
      "CPU": [1, 0],
      "GPU": [0, 1],
    }[label]

TASKS = {
  "GrewePredictive": GrewePredictive,
}
