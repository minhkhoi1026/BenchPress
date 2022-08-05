"""
Log file evaluation.
"""
from deeplearning.benchpress.experiments import public

@public.evaluator
def LogFile(**kwargs) -> None:
  """
  Write benchmarks  and target stats in log file.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')
  raise NotImplementedError
  return