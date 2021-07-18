"""
Feature space sampling of source code.
"""
import typing
import tempfile
import contextlib
import pathlib
import math

from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from eupy.native import logger as l

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "benchmarks_path",
  "./rodinia_benchmarks",
  "Set path of target benchmarks for active sampling."
)

class EuclideanSampler(object):
  """
  This is a shitty experimental class to work with benchmark comparison.
  Will be refactored obviously.
  """
  class Benchmark(typing.NamedTuple):
    path: pathlib.Path
    name: str
    contents: str
    feature_vector: typing.Dict[str, float]

  def __init__(self,
               workspace: pathlib.Path,
               feature_space: str
               ):
    self.path          = pathlib.Path(FLAGS.benchmarks_path).resolve()
    self.workspace     = workspace
    self.feature_space = feature_space
    self.loadCheckpoint()
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.getLogger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.feature_vector))
    except IndexError:
      self.target_benchmark = None
    return

  def iter_benchmark(self):
    """
    When it's time, cycle through the next target benchmark.
    """
    # self.benchmarks.append(self.benchmarks.pop(0))
    try:
      self.target_benchmark = self.benchmarks.pop(0)
    except IndexError:
      self.target_benchmark = None
    self.saveCheckpoint()
    l.getLogger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.feature_vector))
    return

  def calculate_distance(self, infeat: typing.Dict[str, float]) -> float:
    """
    Euclidean distance between sample feature vector
    and current target benchmark.
    """
    d = 0
    for key in self.target_benchmark.feature_vector.keys():
      i = infeat[key]
      t = self.target_benchmark.feature_vector[key]
      d += abs((t**2) - (i**2))
    return math.sqrt(d)

  def topK_candidates(self,
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      K : int,
                      ) -> typing.List[typing.TypeVar("ActiveSample")]:
    """
    Return top-K candidates.
    """
    return sorted(candidates, key = lambda x: x.score)[:K]

  def sample_from_set(self, 
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      search_width: int,
                      ) -> bool:
    """
    Find top K candidates by getting minimum
    euclidean distance from set of rodinia benchmarks.
    """
    """
    for idx in range(len(candidates)):
      candidates[idx] = candidates[idx]._replace(
        score = self.calculate_distance(candidates[idx].features)
      )
    """
    return self.topK_candidates(candidates, search_width)

  def saveCheckpoint(self) -> None:
    """
    Save feature sampler state.
    """
    with open(self.workspace / "feature_sampler_state.pkl", 'wb') as outf:
      pickle.dump(self.benchmarks, outf)
    return


  def loadCheckpoint(self) -> None:
    """
    Load feature sampler state.
    """
    if (self.workspace / "feature_sampler_state.pkl").exists():
      with open(self.workspace / "feature_sampler_state.pkl", 'rb') as infile:
        self.benchmarks = pickle.load(infile)
    else:
      self.benchmarks = []
      with self.GetContentFileRoot() as root:
        contentfiles = []
        for file in root.iterdir():
          with open(file, 'r') as inf:
            contentfiles.appendI((file, inf.read()))
      kernels = [(p, k) for k in opencl.ExtractOnlySingleKernels(cf) for p, cf in contentfiles]
      for p, k in kernels:
        features = extractor.ExtractFeatures(k, [self.feature_space])
        if features[self.feature_space]:
          self.benchmarks.append(
            EuclideanSampler.Benchmark(
                p,
                p.name,
                k,
                features[self.feature_space],
              )
          )
    return

  @contextlib.contextmanager
  def GetContentFileRoot(self) -> pathlib.Path:
    """
    Extract tar archive of benchmarks and yield the root path of all files.

    Yields:
      The path of a directory containing content files.
    """
    with tempfile.TemporaryDirectory(prefix=self.path.stem) as d:
      cmd = [
        "tar",
        "-xf",
        str(self.path),
        "-C",
        d,
      ]
      subprocess.check_call(cmd)
      yield pathlib.Path(d)
