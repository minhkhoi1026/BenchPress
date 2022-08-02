"""
Target benchmark analysis evaluator.
"""
import tqdm
import sklearn
from sklearn.decomposition import PCA

from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.util import plotter

@public.evaluator
def AnalyzeTarget(**kwargs) -> None:
  """
  Analyze requested target benchmark suites.
  """
  targets   = kwargs.get('targets')
  tokenizer = kwargs.get('tokenizer')
  workspace_path = kwargs.get('workspace_path')
  raise NotImplementedError
  return

@public.evaluator
def TokenSizeDistribution(**kwargs) -> None:
  """
  Plot token size distribution among multiple SamplesDatabases.
  """
  db_groups      = kwargs.get('db_groups')
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  names = []
  token_lens = []

  for dbg in db_groups:
    if dbg.db_type != samples_database.SamplesDatabase:
      raise ValueError("Token size distribution requires SamplesDatabase. Received {}".format(dbg.db_type))

    lens = []
    for db in dbg.databases:
      lens += db.get_compilable_num_tokens
    names.append(dbg.group_name)
    token_lens.append(lens)

  plotter.RelativeDistribution(
    x         = names,
    y         = token_lens,
    plot_name = "{}_token_dist".format('-'.join(names)),
    path      = workspace_path,
    x_name    = "Token Length",
    **plot_config if plot_config else {},
  )
  return

@public.evaluator
def LLVMInstCountDistribution(**kwargs) -> None:
  """
  Plot LLVM Instruction count distribution among functions in SamplesDatabase dbs.
  """
  db_groups      = kwargs.get('db_groups')
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  names = []
  token_lens = []

  for dbg in db_groups:
    if dbg.db_type != samples_database.SamplesDatabase:
      raise ValueError("Token size distribution requires SamplesDatabase. Received {}".format(dbg.db_type))

    lens = []
    for db in dbg.databases:
      lens += [x[1]["InstCountFeatures"]["TotalInsts"] for x in db.get_samples_features if "InstCountFeatures" in x[1]]
    names.append(dbg.group_name)
    token_lens.append(lens)

  plotter.RelativeDistribution(
    x         = names,
    y         = token_lens,
    plot_name = "{}_llvm_inst".format('-'.join(names)),
    path      = workspace_path,
    x_name    = "LLVM IR Instructions Length (-O1)"
    **plot_config if plot_config else {},
  )
  return

@public.evaluator
def PCASamplesFeatures(**kwargs) -> None:
  """
  Plot PCA-ed features of different SamplesDatabase samples.
  """
  db_groups      = kwargs.get('db_groups')
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')
  feature_space  = kwargs.get('feature_space')

  indexed_data = {}
  full_data    = []

  scaler = sklearn.preprocessing.StandardScaler()

  i = 0
  for dbg in db_groups:
    if dbg.db_type != samples_database.SamplesDatabase:
      raise ValueError("Token size distribution requires SamplesDatabase. Received {}".format(dbg.db_type))

    ds = []
    for db in dbg.databases:
      ds += [x for _, x in db.get_samples_features if feature_space in x]

    indexed_data[dbg.group_name] = {}
    indexed_data[dbg.group_name]['start'] = i
    for x in ds:
      vals = list(x[feature_space].values())
      if vals:
        i += 1
        full_data.append([float(y) for y in vals])
    
    indexed_data[dbg.group_name]['end'] = i

  # scaled = scaler.fit_transform(full_data)
  reduced = PCA(2).fit_transform(full_data)
  groups = {}
  for dbg in db_groups:
    groups[dbg.group_name] = {
      "names" : [],
      "data"  : reduced[indexed_data[dbg.group_name]['start']: indexed_data[dbg.group_name]['end']],
    }
  plotter.GroupScatterPlot(
    groups = groups,
    title  = "PCA-2 {}".format(feature_space.replace("Features", " Features")),
    plot_name = "pca2_{}_{}".format(feature_space, '-'.join([str(x) for x in groups.keys()])),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  return

@public.evaluator
def FeaturesDistribution(**kwargs) -> None:
  """
  Plot distribution of features per feature dimension per database group.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')
  data = {}

  # You need this if you want to have the same (github) baseline but when github is not plotted.
  reduced_git = None
  for dbg in db_groups:
    if dbg.group_name == "GitHub-768-inactive" or dbg.group_name == "GitHub-768":
      reduced_git = dbg.get_data_features(feature_space)
      break

  radar_groups = {}
  max_fvals    = {}
  benchmarks = target.get_benchmarks(feature_space, reduced_git_corpus = reduced_git)
  for idx, dbg in enumerate(db_groups):
    if dbg.group_name == "GitHub-768-inactive":
      # Skip baseline DB group.
      continue
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles or dbg.db_type == clsmith.CLSmithDatabase):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)
    data[dbg.group_name] = {}
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      # Find shortest distances.
      if unique_code:
        get_data = lambda x: dbg.get_unique_data_features(x)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      if idx == 0:
        if target.target not in data:
          data[target.target] = {}
        for k, v in benchmark.features.items():
          if k not in data[target.target]:
            data[target.target][k] = [v]
          else:
            data[target.target][k].append(v)
          if k not in max_fvals:
            max_fvals[k] = v
          else:
            max_fvals[k] = max(max_fvals[k], v)

      if "{}_{}".format(benchmark.name, feature_space) not in radar_groups:
        radar_groups["{}_{}".format(benchmark.name, feature_space)] = {}
        if target.target not in radar_groups["{}_{}".format(benchmark.name, feature_space)]:
          keys, vals = zip(*sorted(zip(list(benchmark.features.keys()), list(benchmark.features.values()))))
          keys, vals = list(keys), list(vals)
          radar_groups["{}_{}".format(benchmark.name, feature_space)][target.target] = [vals, keys]

      ret = workers.SortedSrcFeatsDistances(get_data(feature_space), benchmark.features, feature_space)[:top_k]
      for _, _, fvec, _ in ret:
        for k, v in fvec.items():
          if k not in data[dbg.group_name]:
            data[dbg.group_name][k] = [v]
          else:
            data[dbg.group_name][k].append(v)
          if k not in max_fvals:
            max_fvals[k] = v
          else:
            max_fvals[k] = max(max_fvals[k], v)

        keys, vals = zip(*sorted(zip(list(fvec.keys()), list(fvec.values()))))
        keys, vals = list(keys), list(vals)
        if dbg.group_name not in radar_groups["{}_{}".format(benchmark.name, feature_space)]:
          radar_groups["{}_{}".format(benchmark.name, feature_space)][dbg.group_name] = [vals, keys]

  plotter.GrouppedViolins(
    data = data,
    plot_name = "feat_distr_{}_dist_{}_{}".format(top_k, feature_space.replace("Features", " Features"), '-'.join([dbg.group_name for dbg in db_groups])),
    path = workspace_path,
    **plot_config if plot_config else {},
  )

  for benchmark, groups in radar_groups.items():
    for k, (values, thetas) in groups.items():
      for idx, (val, theta) in enumerate(zip(values, thetas)):
        if max_fvals[theta] > 0:
          radar_groups[benchmark][k][0][idx] = radar_groups[benchmark][k][0][idx] / max_fvals[theta]
        else:
          radar_groups[benchmark][k][0][idx] = 1.0
    plotter.GrouppedRadar(
      groups,
      plot_name = "radar_{}_{}_{}".format(benchmark, feature_space, '-'.join([dbg.group_name for dbg in db_groups])),
      path      = workspace_path,
      title     = "{}".format(benchmark)
    )
  return

@public.evaluator
def HumanLikeness(**kwargs) -> None:
  """
  Show the top-k candidate codes per target for each DB group,
  to compare how human likely it looks.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')
  data = {}

  # You need this if you want to have the same (github) baseline but when github is not plotted.
  reduced_git = None
  for dbg in db_groups:
    if dbg.group_name == "GitHub-768-inactive" or dbg.group_name == "GitHub-768":
      reduced_git = dbg.get_data_features(feature_space)
      break

  benchmarks = target.get_benchmarks(feature_space, reduced_git_corpus = reduced_git)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
    for idx, dbg in enumerate(db_groups):
      if dbg.group_name == "GitHub-768-inactive":
        # Skip baseline DB group.
        continue
      if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles or dbg.db_type == clsmith.CLSmithDatabase):
        raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)
      # Find shortest distances.
      if unique_code:
        get_data = lambda x: dbg.get_unique_data_features(x)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      if benchmark.name not in data:
        data[benchmark.name] = {
          target.target: benchmark.contents
        }
      data[benchmark.name][dbg.group_name] = [
        src for src, _, _, _
        in workers.SortedSrcFeatsDistances(get_data(feature_space), benchmark.features, feature_space)[:top_k]
      ]
  return
