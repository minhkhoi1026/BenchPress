"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import datetime
import tqdm
import pathlib
import pickle
import copy
import math
import numpy as np

from deeplearning.clgen.models.torch_bert import optimizer
from deeplearning.clgen.models.torch_bert import hooks
from deeplearning.clgen.active_models import backends
from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.active_models.committee import models
from deeplearning.clgen.active_models.committee import config
from deeplearning.clgen.active_models.committee import committee_database
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import logging as l

from absl import flags

FLAGS = flags.FLAGS

class QueryByCommittee(backends.BackendBase):

  class TrainingOpts(typing.NamedTuple):
    """Wrapper class for training options"""
    train_batch_size : int
    learning_rate    : float
    num_warmup_steps : int
    max_grad_norm    : float
    steps_per_epoch  : int
    num_epochs       : int
    num_train_steps  : int
    n_clusters       : int
    init             : str
    n_init           : int
    max_iter         : int
    tol              : float
    algorithm        : str
    n_neighbors      : int
    weights          : str
    leaf_size        : int
    p                : float

  class CommitteeEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : 'torch.utils.data.Dataset'
    optimizer      : typing.Any
    scheduler      : typing.Any
    training_opts  : 'TrainingOpts'
    sha256         : str
    config         : config.ModelConfig
    train_fn       : typing.Callable
    sample_fn      : typing.Callable

  def __repr__(self):
    return "QueryByCommittee"

  def __init__(self, *args, **kwargs):

    super(QueryByCommittee, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.random_seed)
    self.torch.cuda.manual_seed_all(self.config.random_seed)

    self.ckpt_path         = self.cache_path / "checkpoints"
    self.sample_path       = self.cache_path / "samples"
    self.logfile_path      = self.cache_path / "logs"

    self.validation_results_file = "val_results.txt"
    self.validation_results_path = self.logfile_path / self.validation_results_file

    self.committee         = None

    self.is_validated      = False
    self.is_trained        = False

    self.committee_samples = committee_database.CommitteeSamples(
      url        = "sqlite:///{}".format(str(self.sample_path / "samples.db")),
      must_exist = False,
    )
    l.logger().info("Active Committee config initialized in {}".format(self.cache_path))
    return

  def _ConfigModelParams(self,
                         data_generator : 'torch.utils.data.Dataset' = None,
                         is_sampling    : bool = False
                         ) -> None:
    """
    Model parameter initialization.
    """
    if not self.committee:
      self.committee = []
      self.committee_configs = config.ModelConfig.FromConfig(self.config.committee, self.downstream_task)
      for idx, cconfig in enumerate(self.committee_configs):
        training_opts = QueryByCommittee.TrainingOpts(
          train_batch_size = cconfig.batch_size,
          learning_rate    = cconfig.learning_rate,
          num_warmup_steps = cconfig.num_warmup_steps,
          max_grad_norm    = cconfig.max_grad_norm,
          steps_per_epoch  = cconfig.steps_per_epoch,
          num_epochs       = cconfig.num_epochs,
          num_train_steps  = cconfig.num_train_steps,
          n_clusters       = cconfig.n_clusters,
          init             = cconfig.init,
          n_init           = cconfig.n_init,
          max_iter         = cconfig.max_iter,
          tol              = cconfig.tol,
          algorithm        = cconfig.algorithm,
          n_neighbors      = cconfig.n_neighbors,
          weights          = cconfig.weights,
          leaf_size        = cconfig.leaf_size,
          p                = cconfig.p,
        )
        cm = models.CommitteeModels.FromConfig(idx, cconfig)
        if not is_sampling and isinstance(cm, self.torch.nn.Module):
          opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
            model           = cm,
            num_train_steps = 10**5,
            warmup_steps    = training_opts.num_warmup_steps,
            learning_rate   = training_opts.learning_rate,
          )
        else:
          opt, lr_scheduler = None, None
        self.committee.append(
          QueryByCommittee.CommitteeEstimator(
            model          = cm,
            data_generator = data_generator,
            optimizer      = opt,
            scheduler      = lr_scheduler,
            training_opts  = training_opts,
            sha256         = cconfig.sha256,
            config         = cconfig,
            train_fn       = self.TrainNNMember if isinstance(cm, self.torch.nn.Module) else self.TrainUnsupervisedMember,
            sample_fn      = self.SampleNNMember if isinstance(cm, self.torch.nn.Module) else self.SampleUnsupervisedMember,
          )
        )
        (self.ckpt_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
        (self.logfile_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
      l.logger().info(self.GetShortSummary())
    for member in self.committee:
      self.committee_samples.add_member(
        member_id     = member.model.id,
        member_name   = member.config.name,
        type          = "supervised" if isinstance(member.model, self.torch.nn.Module) else "unsupervised",
        configuration = member.config.config,
      )
    return

  def model_step(self,
                 model: 'torch.nn.module',
                 inputs: typing.Dict[str, 'torch.Tensor'],
                 is_sampling: bool = False
                 ) -> float:
    """
    Run forward function for member model.
    """
    outputs = model(
      input_ids   = inputs['input_ids'].to(self.pytorch.device),
      target_ids  = inputs['target_ids'].to(self.pytorch.device) if not is_sampling else None,
      is_sampling = is_sampling,
    )
    return outputs

  def TrainNNMember(self, member: 'QueryByCommittee.CommitteeEstimator', **kwargs) -> None:
    """
    Member-dispatching function for loading checkpoint, training and saving back.
    """
    update_dataloader = kwargs.get('update_dataloader', None)

    model           = member.model.to(self.pytorch.offset_device)
    data_generator  = (
      member.data_generator
      if update_dataloader is None
      else update_dataloader
           + member.data_generator.get_random_subset(
               max(0, abs(len(update_dataloader) - member.training_opts.num_train_steps)))
    )
    optimizer       = member.optimizer
    scheduler       = member.scheduler
    member_path     = self.ckpt_path / member.sha256
    member_log_path = self.logfile_path / member.sha256

    if self.pytorch.num_nodes > 1:
      distrib.barrier()
      model = self.torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids    = [self.pytorch.offset_device],
        output_device = self.pytorch.offset_device,
      )
    elif self.pytorch.num_gpus > 1:
      model = self.torch.nn.DataParallel(model)

    current_step = self.loadCheckpoint(model, member_path, optimizer, scheduler)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    if current_step >= 0:
      l.logger().info("Loaded {} checkpoint step {}".format("{}-{}".format(member.config.name, member.model.id), current_step))
    current_step = max(0, current_step)
    num_train_steps = member.training_opts.num_train_steps if update_dataloader is None else member.training_opts.num_train_steps + current_step

    if current_step < num_train_steps:
      model.zero_grad()

      ## Set batch size in case of TPU training or distributed training.
      if self.torch_tpu_available:
        total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
      else:
        total_train_batch_size = (
          member.training_opts.train_batch_size
          * (self.torch.distributed.get_world_size() if self.pytorch.num_nodes > 1 else 1)
        )

      if self.torch.distributed.get_world_size() <= 1:
        sampler = self.torch.utils.data.RandomSampler(data_generator, replacement = False)
      else:
        sampler = self.torch.utils.data.DistributedSampler(
          data_generator,
          num_replicas = self.pytorch.num_nodes,
          rank         = self.pytorch.torch.distributed.get_rank()
        )
      loader = self.torch.utils.data.dataloader.DataLoader(
        dataset    = data_generator,
        batch_size = member.training_opts.train_batch_size,
        sampler    = (sampler
          if self.pytorch.num_nodes <= 1 or not self.pytorch.torch_tpu_available or self.pytorch.torch_xla.xrt_world_size() <= 1
          else self.torch.utils.data.distributed.DistributedSampler(
            dataset      = data_generator,
            num_replicas = self.pytorch.num_nodes if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.xrt_world_size(),
            rank         = self.pytorch.torch.distributed.get_rank() if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.get_ordinal()
          )
        ),
        num_workers = 0,
        drop_last   = True if environment.WORLD_SIZE > 1 else False,
      )
      # Set dataloader in case of TPU training.
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            data_generator, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)

      # Get dataloader iterator and setup hooks.
      batch_iterator = iter(loader)
      if self.is_world_process_zero():
        train_hook = hooks.tensorMonitorHook(
          member_log_path, current_step, min(member.training_opts.steps_per_epoch, 50)
        )
      try:
        with self.torch.enable_grad():
          model.train()
          # epoch_iter = tqdm.auto.trange(member.training_opts.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(member.training_opts.num_epochs)
          epoch = num_train_steps // member.training_opts.steps_per_epoch
          # In distributed mode, calling the set_epoch() method at
          # the beginning of each epoch before creating the DataLoader iterator
          # is necessary to make shuffling work properly across multiple epochs.
          # Otherwise, the same ordering will be always used.
          if self.pytorch.num_nodes > 1:
            loader.sampler.set_epoch(epoch)

          batch_iter = tqdm.auto.trange(member.training_opts.steps_per_epoch, desc="Batch", leave = False) if self.is_world_process_zero() else range(member.training_opts.steps_per_epoch)
          for step in batch_iter:
            if self.is_world_process_zero():
              start = datetime.datetime.utcnow()
            try:
              inputs = next(batch_iterator)
            except StopIteration:
              # dataloader has different len() than steps_per_epoch.
              # This is the easiest way to infinite-loop dataloaders in pytorch.
              batch_iterator = iter(loader)
              inputs = next(batch_iterator)

            # Run model step on inputs
            step_out = self.model_step(model, inputs)
            # Backpropagate losses
            total_loss = step_out['total_loss'].mean()
            total_loss.backward()

            self.torch.nn.utils.clip_grad_norm_(model.parameters(), member.training_opts.max_grad_norm)
            if self.torch_tpu_available:
              self.pytorch.torch_xla.optimizer_step(optimizer)
            else:
              optimizer.step()
            scheduler.step()

            ## Collect tensors for logging.
            if self.pytorch.num_nodes > 1:
              total_loss = [self.torch.zeros(tuple(step_out['total_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
              self.torch.distributed.all_gather(total_loss, step_out["total_loss"])
            else:
              total_loss = step_out['total_loss'].unsqueeze(0).cpu()
            if self.is_world_process_zero():
              train_hook.step(
                train_step = current_step,
                total_loss = sum([tl.mean().item() for tl in total_loss]) / len(total_loss),
              )
            model.zero_grad()
            if current_step == 0:
              l.logger().info("Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))
            current_step += 1
          # End of epoch
          self.saveCheckpoint(
            model,
            member_path,
            optimizer = optimizer,
            scheduler = scheduler,
            step = current_step
          )
          if self.pytorch.num_nodes > 1:
            loader.sampler.set_epoch(epoch)

          if self.is_world_process_zero():
            l.logger().info("Epoch {} Loss: {}".format(current_step // member.training_opts.steps_per_epoch, train_hook.epoch_loss))
            train_hook.end_epoch()

          if self.torch_tpu_available:
            self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())
      except KeyboardInterrupt:
        pass
    return

  def TrainUnsupervisedMember(self, member: 'QueryByCommittee.CommitteeEstimator', **kwargs) -> None:
    """
    Train non-NeuralNetwork based architectures, such as DecisionTrees or KMeans.
    """
    update_dataloader = kwargs.get('update_dataloader', None)

    model          = member.model
    data_generator = member.data_generator + update_dataloader
    train_dataset  = data_generator.get_batched_dataset()

    member_path     = self.ckpt_path / member.sha256
    member_log_path = self.logfile_path / member.sha256

    current_step = self.loadCheckpoint(model, member_path)
    if current_step >= 0:
      l.logger().info("Loaded {} checkpoint step {}".format("{}-{}".format(member.config.name, member.model.id), current_step))
    if current_step < 0 or update_dataloader is not None:
      current_step = max(0, current_step)
      outputs = model(
        input_ids   = train_dataset['input_ids'],
        target_ids  = train_dataset['target_ids'],
        is_sampling = False,
      )
      self.saveCheckpoint(
        model,
        member_path,
        step = current_step + 1,
      )
    return

  def Train(self, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    # Configure committee members.
    update_dataloader = kwargs.get('update_dataloader', None)

    self._ConfigModelParams(self.downstream_task.data_generator)
    if not self.is_trained or update_dataloader is not None:
      for member in self.committee:
        member.train_fn(member, update_dataloader = update_dataloader)
    self.is_trained = True
    return

  def Validate(self) -> None:
    """
    Perform validation for committee members.
    """
    raise NotImplementedError
    return

  def SampleNNMember(self,
                     member     : 'QueryByCommittee.CommitteeEstimator',
                     sample_set : 'torch.utils.data.Dataset',
                     ) -> typing.Dict[str, typing.List]:
    """
    Sample member of committee. Return predicted label.
    """
    model           = member.model.to(self.pytorch.offset_device)
    member_path     = self.ckpt_path / member.sha256
    member_log_path = self.logfile_path / member.sha256

    if self.pytorch.num_nodes > 1:
      distrib.barrier()
      model = self.torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids    = [self.pytorch.offset_device],
        output_device = self.pytorch.offset_device,
      )
    elif self.pytorch.num_gpus > 1:
      model = self.torch.nn.DataParallel(model)

    current_step = self.loadCheckpoint(model, member_path)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    if current_step >= 0:
      l.logger().info("Loaded {} checkpoint step {}".format("{}-{}".format(member.config.name, member.model.id), current_step))
    current_step = max(0, current_step)

    if self.pytorch.num_nodes <= 1:
      sampler = self.torch.utils.data.SequentialSampler(sample_set)
    else:
      sampler = self.torch.utils.data.DistributedSampler(
        sample_set,
        num_replicas = self.pytorch.num_nodes,
        rank         = self.pytorch.torch.distributed.get_rank(),
        shuffle      = False,
        drop_last    = False,
      )
    loader = self.torch.utils.data.dataloader.DataLoader(
      dataset    = sample_set,
      batch_size = member.training_opts.train_batch_size,
      sampler    = (sampler
        if self.pytorch.num_nodes <= 1 or not self.pytorch.torch_tpu_available or self.pytorch.torch_xla.xrt_world_size() <= 1
        else self.torch.utils.data.distributed.DistributedSampler(
          dataset      = sample_set,
          num_replicas = self.pytorch.num_nodes if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.xrt_world_size(),
          rank         = self.pytorch.torch.distributed.get_rank() if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.get_ordinal()
        )
      ),
      num_workers = 0,
      drop_last   = False # True if environment.WORLD_SIZE > 1 else False,
    )
    # Set dataloader in case of TPU training.
    if self.torch_tpu_available:
      loader = self.pytorch.torch_ploader.ParallelLoader(
                          sample_set, [self.pytorch.device]
                        ).per_device_loader(self.pytorch.device)
    # Get dataloader iterator and setup hooks.
    model.eval()
    predictions = {
      'train_step'      : current_step,
      'idx'             : None,
      'static_features' : None,
      'runtime_features': None,
      'input_ids'       : None,
      'predictions'     : None,
    }
    for batch in tqdm.tqdm(loader, total = len(loader), desc = "Sample member", leave = False):
      out = self.model_step(model, batch, is_sampling = True)
      for key in set(predictions.keys()) - set({'train_step'}):
        r = batch[key] if key != "predictions" else out['output_label']
        if predictions[key] is None:
          predictions[key] = r
        else:
          predictions[key] = self.torch.cat(
            (predictions[key], r),
            0
          )

    if self.pytorch.num_nodes > 1:
      self.torch.distributed.barrier()

      idx              = [self.torch.zeros(tuple(predictions['idx'].shape),              dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      static_features  = [self.torch.zeros(tuple(predictions['static_features'].shape),  dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      runtime_features = [self.torch.zeros(tuple(predictions['runtime_features'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      input_ids        = [self.torch.zeros(tuple(predictions['input_ids'].shape),        dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      output_label     = [self.torch.zeros(tuple(predictions['predictions'].shape),      dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]

      self.torch.distributed.all_gather(idx,              predictions["idx"].to(self.pytorch.device))
      self.torch.distributed.all_gather(static_features,  predictions["static_features"].to(self.pytorch.device))
      self.torch.distributed.all_gather(runtime_features, predictions["runtime_features"].to(self.pytorch.device))
      self.torch.distributed.all_gather(input_ids,        predictions["input_ids"].to(self.pytorch.device))
      self.torch.distributed.all_gather(output_label,     predictions["predictions"])

      predictions['idx']              = self.torch.cat(idx)
      predictions['static_features']  = self.torch.cat(static_features)
      predictions['runtime_features'] = self.torch.cat(runtime_features)
      predictions['input_ids']        = self.torch.cat(input_ids)
      predictions['predictions']      = self.torch.cat(output_label)

    for key in set(predictions.keys()) - set({'train_step'}):
      if key == 'predictions':
        predictions[key] = [self.downstream_task.TargetIDtoLabels(int(x)) for x in predictions[key].cpu().numpy()]
      elif key == "runtime_features":
        predictions[key] = [[int(y) for y in x.cpu().numpy()] for x in predictions[key]]
      elif key == "idx":
        predictions[key] = [int(x.cpu().numpy()) for x in predictions[key]]
      else:
        predictions[key] = [[float(y) for y in x.cpu().numpy()] for x in predictions[key]]
    return predictions

  def SampleUnsupervisedMember(self,
                               member     : 'QueryByCommittee.CommitteeEstimator',
                               sample_set : 'torch.utils.data.Dataset',
                               ) -> typing.Dict[str, typing.List]:
    """
    Sample non-NeuralNetwork based architectures, such as DecisionTrees or KMeans.
    """
    model          = member.model
    sample_dataset = sample_set.get_batched_dataset()

    member_path     = self.ckpt_path / member.sha256
    member_log_path = self.logfile_path / member.sha256

    current_step = self.loadCheckpoint(model, member_path)
    if current_step >= 0:
      l.logger().info("Loaded {} checkpoint step {}".format("{}-{}".format(member.config.name, member.model.id), current_step))
    current_step = max(0, current_step)

    outputs = model(
      input_ids   = sample_dataset['input_ids'],
      is_sampling = True,
    )
    predictions = {
      'train_step'      : current_step,
      'idx'             : [int(x) for x in sample_dataset['idx']],
      'static_features' : sample_dataset['static_features'],
      'runtime_features': sample_dataset['runtime_features'],
      'input_ids'       : sample_dataset['input_ids'],
      'predictions'     : [self.downstream_task.TargetIDtoLabels(i) for i in outputs['predicted_labels']],
    }
    return predictions

  def SampleCommittee(self,
                      sample_set: 'torch.utils.data.Dataset',
                      ) -> typing.Dict[
                             'QueryByCommittee.CommitteeEstimator',
                             typing.Dict[str, 'torch.Tensor']
                            ]:
    """
    Sample committee with a set of inputs.
    Return a dictionary mapped from each member to the
    total workload computed by a committee member.
    """
    self._ConfigModelParams()
    committee_predictions = {}
    for member in self.committee:
      key = "{}_{}".format(member.config.name, member.model.id)
      committee_predictions[key] = member.sample_fn(member, sample_set)
    return committee_predictions

  def Sample(self, num_samples: int = 512) -> typing.List[typing.Dict[str, float]]:
    """
    Active learner sampling.
    This method queries all committee members and measures their cross-entropy to validate
    the usefulness of parts of the feature space.
    """
    # Collect random space samples.
    sample_set            = self.downstream_task.sample_space(num_samples = 512)
    # Ask the committee for their predictions.
    committee_predictions = self.SampleCommittee(sample_set)
    space_samples = []
    for nsample in range(len(sample_set)):
      # Get the feature vectors for each sample.
      for model, samples in committee_predictions.items():
        if nsample != samples['idx'][nsample]:
          raise ValueError("{} Mismatch in sample output: Expected {} but had {}".format(model, nsample, samples['idx'][nsample]))
        static_feats = self.downstream_task.VecToStaticFeatDict(samples['static_features'][nsample])
        run_feats    = self.downstream_task.VecToRuntimeFeatDict(samples['runtime_features'][nsample])
        input_feats  = self.downstream_task.VecToInputFeatDict(samples['input_ids'][nsample])
        break
      # Calculate entropy for that sample.
      ent = self.entropy([x['predictions'][nsample] for x in committee_predictions.values()])
      # Save the dictionary entry.
      space_samples.append({
        'train_step'         : {k: v['train_step'] for k, v in committee_predictions.items()},
        'static_features'    : static_feats,
        'runtime_features'   : run_feats,
        'input_features'     : input_feats,
        'member_predictions' : {k: v['predictions'][nsample] for k, v in committee_predictions.items()},
        'entropy'            : ent,
      })
    # Add everything to database.
    self.committee_samples.add_samples(space_samples)
    return sorted(space_samples, key = lambda x: x['entropy'], reverse = True)

  def entropy(self, labels, base=None):
    """ Computes entropy of label distribution. """
    if len(labels) <= 1:
      return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
      return 0

    entropy = 0.0
    # Compute entropy
    base = math.e if base is None else base
    for p in probs:
      entropy -= p * math.log(p, base)
    return entropy

  def saveCheckpoint(self, 
                     model : 'torch.nn.Module',
                     path  : pathlib.Path,
                     optimizer  = None,
                     scheduler  = None,
                     step : int = None,
                     ) -> None:
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda x: path / "{}-{}.pt".format(x, step)

      if isinstance(model, self.torch.nn.Module):
        if self.torch_tpu_available:
          if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
            self.pytorch.torch_xla_model.save(model, ckpt_comp("model"))
          self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
          self.pytorch.torch_xla.save(optimizer.state_dict(), ckpt_comp("optimizer"))
          self.pytorch.torch_xla.save(scheduler.state_dict(), ckpt_comp("scheduler"))
        else:
          if isinstance(model, self.torch.nn.DataParallel):
            self.torch.save(model.module.state_dict(), ckpt_comp("model"))
          else:
            self.torch.save(model.state_dict(), ckpt_comp("model"))
          self.torch.save(optimizer.state_dict(), ckpt_comp("optimizer"))
          self.torch.save(scheduler.state_dict(), ckpt_comp("scheduler"))
      else:
        checkpoint_dict = model.get_checkpoint_state()
        with open(ckpt_comp("model"), 'wb') as outf:
          pickle.dump(checkpoint_dict, outf)

      with open(path / "checkpoint.meta", 'a') as mf:
        mf.write("train_step: {}\n".format(step))
    return

  def loadCheckpoint(self,
                     model : 'torch.nn.Module',
                     path  : pathlib.Path,
                     optimizer = None,
                     scheduler = None
                     ) -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (path / "checkpoint.meta").exists():
      return -1

    with open(path / "checkpoint.meta", 'r') as mf:
      key     = "train_step"
      get_step  = lambda x: int(x.replace("\n", "").replace("{}: ".format(key), ""))
      lines     = mf.readlines()
      entries   = set({get_step(x) for x in lines if key in x})
    if FLAGS.select_checkpoint_step == -1:
      ckpt_step = max(entries)
    else:
      raise ValueError("{} not found in checkpoint folder.".format(FLAGS.select_checkpoint_step))

    ckpt_comp = lambda x: path / "{}-{}.pt".format(x, ckpt_step)

    if isinstance(model, self.torch.nn.DataParallel):
      try:
        model.module.load_state_dict(
          self.torch.load(ckpt_comp("model"))
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.module.load_state_dict(new_state_dict)
      model.eval()
    elif isinstance(model, self.torch.nn.Module):
      try:
        model.load_state_dict(
          self.torch.load(ckpt_comp("model"), map_location=self.pytorch.device)
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
      model.eval()
    else:
      checkpoint_dict = pickle.load(open(ckpt_comp("model"), 'rb'))
      model.load_checkpoint_state(checkpoint_dict)
    return ckpt_step

  def is_world_process_zero(self) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on
    several machines, this is only going to be :obj:`True` for one process).
    """
    if self.torch_tpu_available:
      return self.pytorch.torch_xla_model.is_master_ordinal(local=False)
    elif self.pytorch.num_nodes > 1:
      return self.torch.distributed.get_rank() == 0
    else:
      return True

  def GetShortSummary(self) -> str:
    return "Short summary TODO"
