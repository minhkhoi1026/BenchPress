<p align="center">
  <br>
<img src="https://github.com/fivosts/clgen/blob/master/docs/logo.png" width="600px" />
<br>
</p>

***

:orange_book:  [__BenchPress: A Deep Active Benchmark Generator__, *PACT 2022*.](https://dl.acm.org/doi/10.1145/3559009.3569644)

__BenchPress__ is a directed program synthesizer for compiler benchmarks. Using active learning, it ranks compiler features based on their significance and produces executables that target them. __BenchPress__ is very accurate in generating compilable code - 87% of single-shot samples from empty feeds compile - while remaining light-weight compared to other massive seq2seq generative models.

## Quick Start

### Setup docker image
You can fetch the docker image from `ghcr.io/minhkhoi1026/benchpress/benchpress:latest` 
```bash
docker pull ghcr.io/minhkhoi1026/benchpress/benchpress:latest
docker tags ghcr.io/minhkhoi1026/benchpress/benchpress:latest benchpress
```

or build it using the `Dockerfile` in `docker` folders:
```bash
cd docker
docker build --tag benchpress --network host --force-rm --no-cache .
```

### Run image
You can run the image to start using BenchPress.
```bash
docker run -itd --rm --gpus all benchpress
```

This command run the container with tag `benchpress`, where `-it` options keep the container alive without running process, `-d` run it in detach mode (not be attached to the parent process so that it can work independent). The last option `--gpus all` specify that the container will have the permission to access all gpus in the host computer. Note that gpu option is only available if you have install the nvidia-container-toolkit [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

If you pull the image from `ghcr`, it will likely it is the old version, so you will have to run these command **inside the container** to fix the issue:
```bash
# install missing packages (ip, tmux, vim)
apt update && apt upgrade
apt install net-tools vim net-tools

# reinstall cuda-compiled version of torch
export BENCHPRESS_BINARY=cmd
./benchpress -m pip  install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

There might be the lack of the `from_pretrained_custom.py` file in the source code, so you have to copy its content from `deeplearning/benchpress/models/from_pretrained_custom.py` to a same file in docker container:
```bash
vim deeplearning/benchpress/models/from_pretrained_custom.py
<paste the code, save and exit>
```

### Run a pretrained-model with custom behaviour
The custom version add the feature to save generated kernels to text file, inside the docker container:
```
$: export BENCHPRESS_BINARY=cmd
$: ./benchpress

>>> from deeplearning.benchpress.models.from_pretrained_custom import PreTrainedModel
>>> pretrained = PreTrainedModel.FromID("base_opencl")
>>> BATCH_SIZE = 8 # sample per batch per gpu
>>> N_GPUS = 4 # change base on the number of GPU in your host
>>> N_BATCHES = 10000 # number of batch
>>> # produce N_BATCHS * BATCH_SIZE * N_GPUS kernels
>>> texts, samples = pretrained.Sample(
    "kernel void [HOLE]}",
    batch_size=BATCH_SIZE,
    num_batches=N_BATCHES,
    sample_workload_size=N_GPUS,
    print_samples=False,
    seed=2610, # change this seed to generate different set of kernels
)
```


### Run a pre-trained model

You want to see some __BenchPress__ samples fast ? You can fetch and run a pre-trained model and experiment with your own string prompts.

```
$: export BENCHPRESS_BINARY=cmd
$: ./benchpress

>>> from deeplearning.benchpress.models.from_pretrained import PreTrainedModel
>>> pretrained = PreTrainedModel.FromID("base_opencl")
>>> texts, samples = pretrained.Sample("kernel void [HOLE]}")
>>> for text in texts:
...   print(text)
>>> help(pretrained.Sample) # To list all Sample parameters.
```

### Training a model

After installing __BenchPress__ you can train your own model on a preset of C/OpenCL corpuses or using your own corpus. See all different model flavors in `model_zoo/BERT/`. To train, run

```
./benchpress --config model_zoo/BERT/online.pbtxt --workspace_dir <your_workspace_directory>
```

__BenchPress__ supports CPU, single node - single GPU, single node - multi GPU and multi node - multi GPU training and inference.

To see all available flags run `./benchpress --help/--helpfull`. Some relevant flags may be:

- `--sample_per_epoch` Set test sampling rounds per epoch end.
- `--validate_per_epoch` Similar to previous.
- `--local_filesystem` Set desired temporary directory. By default `/tmp/` is set.

## Installation

See `INSTALL.md` for instructions.

## Evaluate the code

If you have trained __BenchPress__ and ran a sampler to any downstream task you want to evaluate, you can use the codebase's evaluators. The evaluators usually take a list of database groups and perform operations/analysis/plotting on them. Evaluators are described in protobuf files (see examples in `model_zoo/evaluation/`). To run an evaluator run

```
$: export BENCHPRESS_BINARY=deeplearning/benchpress/experiments/evaluators.py
$: ./benchpress --evaluator_config <path/to/your/evaluator.pbxt>
```

## Github and BigQuery mining

__BenchPress__ provides modules to scrape source code from Github and store it into databases. Language specifications are set through protobuf files. See `model_zoo/github` for examples. For example

```
./benchpress --config model_zoo/github/bq_C_db.pbtxt
```
to scrape C repositories from BigQuery.

__BenchPress__ comes with two datasets. A dataset of ~64,000 OpenCL kernels and a C dataset of ~6.5 million source files (about ~90 million functions). The OpenCL dataset is downloaded automatically if requested through a model description protobuf (see corpus field). The C database doesn't due to its size. If you are interested in it, get in touch.

## Utilities

A range of useful ML utilities reside within __BenchPress's__ codebase that you may find useful. Inside `deeplearning/benchpress/util` you will find standalone modules such as:

- `plotter.py`: A plotly interface that easily plots lines, scatters, radars, bars, groupped bars, stacked bars, histograms, distributions etc.
- `distrib.py`: A utility module that handles distributed environments: barrier(), lock(), unlock(), broadcast_messages() etc.
- `memory.py` : A RAM and GPU memory live tracker and plotter of your application.
- `gpu.py`: Wrapper over `nvidia-smi` for GPU info.
- `monitors.py`: A set of classes that monitor streaming data, store and plot.
- `distributions.py`: Class for distribution operations. Populate distributions and do operations on them (+, -, /, *) and plot PMFs, PDFs.
- `logging.py`: logging module with pretty colors.
- and others!


Share your ideas with me: <fivosts@gmail.com>
