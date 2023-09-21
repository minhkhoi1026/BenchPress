from deeplearning.benchpress.models.from_pretrained_custom import PreTrainedModel


pretrained = PreTrainedModel.FromID("base_opencl")

BATCH_SIZE = 8
N_GPUS = 4
N_BATCHES = 1000

# produce N_BATCHS * BATCH_SIZE * N_GPUS kernels

texts, samples = pretrained.Sample(
    "kernel void [HOLE]}",
    batch_size=BATCH_SIZE,
    num_batches=N_BATCHES,
    sample_workload_size=N_GPUS,
    print_samples=False,
)
