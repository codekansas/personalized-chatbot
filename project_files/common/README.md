# <name>

<description>

## Getting Started

Set some environment variables:

```bash
export RUN_DIR=/path/to/run/dir  # This is where all your training logs and checkpoints will be written
export EVAL_RUN_DIR=/path/to/eval/run/dir  # This is where all your evaluation logs will be written
```

It's also a good idea to set these variables:

```bash
export DATA_DIR=/path/to/data/dir  # This is where your datasets are stored
export MODEL_DIR=/path/to/model/dir  # This is where your pretrained models are stored
```

Check out the documentation [here](https://ml.bolte.cc/getting_started.html).

## Usage

Train a ResNet18 model on CIFAR10:

```bash
runml train configs/image_demo.yaml
```

Train an RL PPO model on BipedalWalker:

```bash
runml train configs/rl_demo.yaml
```

Launch a Slurm job (requires setting the `SLURM_PARTITION` environment variable):

```bash
runml launch configs/image_demo.yaml launcher.name=slurm launcher.num_nodes=1 launcher.gpus_per_node=1
```
