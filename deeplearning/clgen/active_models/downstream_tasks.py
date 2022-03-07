"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib
import tqdm
import multiprocessing

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.features import extractor

def ExtractorWorker(cldrive_entry: cldrive.CLDriveSample, fspace: str):
  """
  Worker that extracts features and buffers cldrive entry, to maintain consistency
  among multiprocessed data.
  """
  features = extractors.ExtractFeatures(cldrive_entry.source, [fspace])
  if fspace in features and features[fspace]:
    return features, cldrive_entry
  return None

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str, corpus_path: pathlib.Path) -> "DownstreamTask":
    return TASKS[task](corpus_path)

  def __init__(self, name) -> None:
    self.name = name
    return

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  This class is responsible to fetch the raw data and act as a tokenizer
  for the data. Reason is, the data generator should be agnostic of the labels.
  """
  def __init__(self, corpus_path: pathlib.Path) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive")
    self.inputs         = ["comp", "mem", "localmem", "coalesced", "atomic"]
    self.input_size     = 10
    self.output_labels  = ["CPU", "GPU"]
    self.output_size    = 2
    self.setup_dataset()
    self.data_generator = data_generator.Dataloader(self.dataset)
    return

  def setup_dataset(self) -> None:
    """
    Fetch data and preprocess into corpus for Grewe's predictive model.
    """
    self.dataset = []
    self.corpus_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus_path)), must_exist = True)
    data    = [x for x in self.corpus_db.get_valid_data()]
    pool = multiprocessing.Pool()
    it = pool.imap_unordered(functools.partial(ExtractorWorker, fspace = "GreweFeatures"), data)
    for dp in tqdm.tqdm(it, total = len(data), desc = "Grewe corpus setup", leave = False):
      if dp:
        feats, entry = dp
        self.dataset.append(
          (
            self.InputtoEncodedVector(feats, entry.transferred_bytes, dp.global_size),
            self.TargetLabeltoEncodedVector(entry.status)
          )
        )
    return

  def InputtoEncodedVector(self,
                           static_feats      : typing.Dict[str, float],
                           transferred_bytes : int,
                           global_size       : int,
                           ) -> typing.List[float]:
    """
    Encode consistently raw features to Grewe's predictive model inputs.
    """
    return [
      transferred_bytes         / (static_feats['comp'] + static_feats['mem']),
      static_feats['coalesced'] / static_feats['mem'],
      static_feats['localmem']  / (static_feats['mem'] * global_size),
      transferred_bytes         / (static_feats['comp'] + static_feats['mem']),
      static_feats['comp']      / static_feats['mem']
    ]

  def TargetIDtoLabels(id: int) -> str:
    """
    Integer ID to label of predictive model.
    """
    return {
      0: "CPU",
      1: "GPU",
    }[id]

  def TargetLabeltoID(label: str) -> int:
    """
    Predictive label to ID.
    """
    return {
      "CPU": 0,
      "GPU": 1,
    }

  def TargetLabeltoEncodedVector(label: str) -> typing.List[int]:
    """
    Label to target vector.
    """
    return {
      "CPU": [1, 0],
      "GPU": [0, 1],
    }

TASKS = {
  "GrewePredictive": GrewePredictive,
}
