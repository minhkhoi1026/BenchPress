"""
Evaluation script for mem vs comp Grewe features against multiple database groups.
"""
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import plotter
from deeplearning.clgen.experiments import public

@public.evaluator
def CompMemGrewe(**kwargs) -> None:
  """
  Compare Computation vs Memory instructions for each database group
  and target benchmarks.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  workspace_path = kwargs.get('workspace_path')
  plot_config    = kwargs.get('plot_config')
  feature_space  = "GreweFeatures"

  groups = {}
  for dbg in db_groups:
    if not isinstance(dg.db_type, samples_database.SamplesDatabase):
      raise ValueError("CompMemGrewe requires SamplesDatabase but received", dbg.db_type)
    groups[dbg.group_name] = {
      'data'  : [],
      'names' : []
    }

  for b in target.get_benchmarks(feature_space):
    groups[target.target]['data'].append([b.features['comp'], b.features['mem']])
    groups[target.target]['names'].append(b.name)
  
  unique = set()
  for dbg in db_groups:
    for feats in dbg.get_features(feature_space):
      if "{}-{}".format(feats['comp'], feats['mem']) not in unique:
        groups[dbg.group_name]["data"].append([feats['comp'], feats['mem']])
        groups[dbg.group_name]['names'].append("")
        unique.add("{}-{}".format(feats['comp'], feats['mem']))

  plotter.GroupScatterPlot(
    groups = groups,
    plot_name = "comp_vs_mem_grewe",
    path = workspace_path,
    **plot_config if plot_config else {}
  )
  return