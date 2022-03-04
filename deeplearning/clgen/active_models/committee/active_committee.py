"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import pathlib
import copy

from deeplearning.clgen.models.torch_bert import optimizer
from deeplearning.clgen.active_models import backends
from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.active_models.committee import models
from deeplearning.clgen.active_models.committee import config
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import logging as l

class ActiveCommittee(backends.BackendBase):

  class TrainingOpts(typing.NamedTuple):
    """Wrapper class for training options"""
    train_batch_size : int
    learning_rate    : float
    num_warmup_steps : int
    max_grad_norm    : float
    steps_per_epoch  : int
    num_epochs       : int
    num_train_steps  : int

  class CommitteeEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : data_generator.Dataloader
    optimizer      : typing.Any
    scheduler      : typing.Any
    training_opts  : 'TrainingOpts'
    sha256         : str

  class SampleCommitteeEstimator(typing.NamedTuple):
    """Named tuple for sampling BERT."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : data_generator.Dataloader
    sha256         : str

  def __repr__(self):
    return "ActiveCommittee"

  def __init__(self, *args, **kwargs):

    super(ActiveCommittee, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.committee.random_seed)
    self.torch.cuda.manual_seed_all(self.config.committee.random_seed)

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"
    self.logfile_path      = self.cache.path / "logs"

    self.validation_results_file = "val_results.txt"
    self.validation_results_path = self.logfile_path / self.validation_results_file

    self.committee         = []

    self.is_validated      = False
    self.trained           = False
    l.logger().info("Active Committee config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, data_generator: data_generator.Dataloader) -> None:
    """
    Model parameter initialization.
    """

    self.committee_configs = config.ModelConfig.FromConfig(self.config.committee, self.downstream_task)
    for idx, cconfig in enumerate(self.committee_configs):
      training_opts = ActiveCommittee.TrainingOpts(
        train_batch_size = cconfig.batch_size,
        learning_rate    = cconfig.learning_rate,
        num_warmup_steps = cconfig.num_warmup_steps,
        max_grad_norm    = cconfig.max_grad_norm,
        steps_per_epoch  = cconfig.steps_per_epoch,
        num_epochs       = cconfig.num_epochs,
        num_train_steps  = cconfig.num_train_steps,
      )
      cm = models.CommitteeModels.FromConfig(idx, cconfig)
      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = cm,
        num_train_steps = training_opts.num_train_steps,
        warmup_steps    = training_opts.num_warmup_steps,
        learning_rate   = training_opts.learning_rate,
      )
      self.committee.append(
        ActiveCommittee.CommitteeEstimator(
          model          = cm,
          data_generator = data_generator,
          optimizer      = opt,
          scheduler      = lr_scheduler,
          training_opts  = training_opts,
          sha256         = cconfig.sha256,
        )
      )
      (self.ckpt_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
      (self.logfile_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
    l.logger().info(self.GetShortSummary())
    return

  def TrainMember(self, member: 'ActiveCommittee.CommitteeEstimator') -> None:
    """
    Member-dispatching function for loading checkpoint, training and saving back.
    """
    model           = member.model.to(self.pytorch.offset_device)
    dataloader      = member.dataloader
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

    current_step = self.loadCheckpoint(model, optimizer, scheduler, member_path)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()

    if current_step >= 0:
      l.logger().info("Loaded checkpoint step {}".format(current_step))

    if current_step < member.num_train_steps:
      model.zero_grad()

      ## Set batch size in case of TPU training or distributed training.
      if self.torch_tpu_available:
        total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
      else:
        total_train_batch_size = (
          member.train_batch_size
          * (self.torch.distributed.get_world_size() if self.pytorch.num_nodes > 1 else 1)
        )

      raise NotImplementedError("Right here implement the loader, sampler etc. thing.")

      # Set dataloader in case of TPU training.
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            self.train.data_generator.dataloader, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)

      # Get dataloader iterator and setup hooks.
      batch_iterator = iter(loader)
      if self.is_world_process_zero():
        train_hook = hooks.tensorMonitorHook(
          member_log_path, current_step, min(steps_per_epoch, 50)
        )
      total_steps = member.num_train_steps
      try:
        model.train()
        epoch_iter = tqdm.auto.trange(member.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(member.num_epochs)
        for epoch in epoch_iter:
          # In distributed mode, calling the set_epoch() method at
          # the beginning of each epoch before creating the DataLoader iterator
          # is necessary to make shuffling work properly across multiple epochs.
          # Otherwise, the same ordering will be always used.
          if self.pytorch.num_nodes > 1:
            loader.sampler.set_epoch(epoch)

          if epoch < current_step // member.steps_per_epoch:
            continue

          batch_iter = tqdm.auto.trange(member.steps_per_epoch, desc="Batch", leave = False) if self.is_world_process_zero() else range(member.steps_per_epoch)
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

            current_step += 1
            # Run model step on inputs
            step_out = self.model_step(model, inputs)
            # Backpropagate losses
            total_loss = step_out['total_loss'].mean()
            total_loss.backward()

            self.torch.nn.utils.clip_grad_norm_(model.parameters(), member.max_grad_norm)
            if self.torch_tpu_available:
              self.pytorch.torch_xla.optimizer_step(optimizer)
            else:
              optimizer.step()
            scheduler.step()

            ## Collect tensors for logging.
            if self.pytorch.num_nodes > 1:
              total_loss = [self.torch.zeros(tuple(step_out['total_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
              self.torch.distributed.all_gather(masked_lm_loss, step_out["masked_lm_loss"])
            else:
              total_loss = step_out['total_loss'].unsqueeze(0).cpu()
            if self.is_world_process_zero():
              train_hook.step(
                train_step = current_step,
                total_loss = total_loss
              )
            model.zero_grad()
            if current_step == 0:
            l.logger().info("Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))
          # End of epoch
          self.saveCheckpoint(model, optimizer, scheduler, member_path)
          if self.pytorch.num_nodes > 1:
            loader.sampler.set_epoch(epoch)

          if self.is_world_process_zero():
            l.logger().info("Epoch {} Loss: {}".format(member.current_step // member.steps_per_epoch, train_hook.epoch_loss))
            train_hook.end_epoch()

          if self.torch_tpu_available:
            self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())
      except KeyboardInterrupt:
        pass
    return

  def Train(self, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    # Configure committee members.
    self._ConfigModelParams(self.downstream_task.data_generator)
    for member in self.committee:
      TrainMember(member)
    return

  def Validate(self) -> None:
    """
    Perform validation for committee members.
    """
    raise NotImplementedError
    return

  def Sample(self) -> None:
    """
    Sample committee members.

    Collect predictions for given inputs of downstream task.
    """
    raise NotImplementedError
    return

  def saveCheckpoint(self, model, optimizer, scheduler, path) -> None:
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, self.current_step)

      if self.torch_tpu_available:
        if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
          self.pytorch.torch_xla_model.save(estimator.model, ckpt_comp("model"))
        self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
        self.pytorch.torch_xla.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
        self.pytorch.torch_xla.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))
      else:
        if isinstance(estimator.model, self.torch.nn.DataParallel):
          self.torch.save(estimator.model.module.state_dict(), ckpt_comp("model"))
        else:
          self.torch.save(estimator.model.state_dict(), ckpt_comp("model"))
        self.torch.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
        self.torch.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))

      with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
        mf.write("train_step: {}\n".format(self.current_step))
    return

  def loadCheckpoint(self, model, optimizer, scheduler, path) -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      key     = "train_step"
      get_step  = lambda x: int(x.replace("\n", "").replace("{}: ".format(key), ""))
      lines     = mf.readlines()
      entries   = set({get_step(x) for x in lines if key in x})
    if FLAGS.select_checkpoint_step == -1:
      ckpt_step = max(entries)
    else:
      if FLAGS.select_checkpoint_step in entries:
        ckpt_step = FLAGS.select_checkpoint_step
      else:
        raise ValueError("{} not found in checkpoint folder.".format(FLAGS.select_checkpoint_step))

    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, ckpt_step)

    if isinstance(estimator.model, self.torch.nn.DataParallel):
      try:
        estimator.model.module.load_state_dict(
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
        estimator.model.module.load_state_dict(new_state_dict)
    else:
      try:
        estimator.model.load_state_dict(
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
        estimator.model.load_state_dict(new_state_dict)
    if isinstance(estimator, ActiveCommittee.CommitteeEstimator):
      if estimator.optimizer is not None and estimator.scheduler is not None and ckpt_step > 0:
        estimator.optimizer.load_state_dict(
          self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device)
        )
        estimator.scheduler.load_state_dict(
          self.torch.load(ckpt_comp("scheduler"), map_location=self.pytorch.device)
        )
    estimator.model.eval()
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
    return "TODO"
