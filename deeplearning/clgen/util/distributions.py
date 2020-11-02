"""Statistical distributions used for sampling"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import plotter

class Distribution():
  def __init__(self, 
               sample_length: int, 
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str
               ):
    self.sample_length  = sample_length
    self.log_path       = log_path if isinstance(log_path, pathlib.Path) else pathlib.Path(log_path)
    self.set_name       = set_name
    self.sample_counter = {}
    return

  @classmethod
  def FromHoleConfig(cls, 
                     config: model_pb2.Hole,
                     log_path: typing.Union[pathlib.Path, str],
                     set_name: str,
                     ):
    if config.HasField("uniform_distribution"):
      return UniformDistribution(config.hole_length,
                                 log_path,
                                 set_name,
                                 )
    elif config.HasField("normal_distribution"):
      return NormalDistribution(config.hole_length, 
                                config.normal_distribution.mean, 
                                config.normal_distribution.variance,
                                log_path,
                                set_name,
                                )
    else:
      raise NotImplementedError(config)

  def sample(self):
    raise NotImplementedError

  def register(self, actual_sample):
    if isinstance(actual_sample, list):
      for s in actual_sample:
        self.register(s)
    else:
      if actual_sample not in self.sample_counter:
        self.sample_counter[actual_sample] =  1
      else:
        self.sample_counter[actual_sample] += 1
    return

  def plot(self):
    sorted_dict = sorted(self.sample_counter.items(), key = lambda x: x[0])
    plotter.FrequencyBars(
      x = [x for (x, _) in sorted_dict],
      y = [y for (_, y) in sorted_dict],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.log_path
    )
    return

class PassiveMonitor(Distribution):
  """
  Not an actual sampling distribution.
  This subclass is used to register values of a specific type, keep track of them
  and bar plot them. E.g. length distribution of a specific encoded corpus.
  """
  def __init__(self, 
               log_path: typing.Union[pathlib.Path, str], 
               set_name: str,
               ):
    super(PassiveMonitor, self).__init__(None, log_path, set_name)
    return

class TimestampMonitor(Distribution):
  """
  Not an actual sampling distribution.
  In contrast to the rest, this subclass does not count frequency of a certain value.
  It registers values and plots them against time.
  """
  def __init__(self, 
               log_path: typing.Union[pathlib.Path, str], 
               set_name: str,
               ):
    super(TimestampMonitor, self).__init__(None, log_path, set_name)
    self.sample_list = []
    return

  def register(self, actual_sample: int) -> None:
    # if isinstance(actual_sample, str):
    #   actual_sample = int(actual_sample)
    self.sample_list.append(float(actual_sample))
    return

  def plot(self):
    plotter.SingleScatterLine(
      x = np.arange(len(self.sample_list)),
      y = self.sample_list,
      title = self.set_name,
      x_name = "",
      y_name = self.set_name,
      plot_name = self.set_name,
      path = self.log_path,
    )
    return

class UniformDistribution(Distribution):
  """
  A uniform distribution sampler. Get a random number from distribution calling sample()
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self, 
               sample_length: int,
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str
               ):
    super(UniformDistribution, self).__init__(sample_length, log_path, set_name)

  def sample(self):
    return np.random.randint(0, self.sample_length + 1)

class NormalDistribution(Distribution):
  """
  Normal distribution sampler. Initialized with mean, variance.
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self,
               sample_length: int,
               mean         : float,
               variance     : float,
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str,
               ):
    super(NormalDistribution, self).__init__(sample_length, log_path, set_name)
    self.mean     = mean
    self.variance = variance

  def sample(self):
    sample = int(round(np.random.normal(loc = self.mean, scale = self.variance)))
    while sample < 0 or sample > self.sample_length:
      sample = int(round(np.random.normal(loc = self.mean, scale = self.variance)))
    return sample
