# Unstable Adventure

An attempt at making a dynamic text adventure with images using StabilityAI models.

## System Requirements

- 3090 / 4090 or 24GB+ Dedicated GPU Memory
- 48GB of System Memory ( RAM )
- 60GB Free Disk Space ( For models, libraries and output )
- Decent Modern Processor
- Good internet to download models

## Installation

### Install Conda:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Create environment with pytorch:

```
conda create --name unstable python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Activate the environment:

```
activate unstable
```

### Install essential packages:

```
python -m pip install --upgrade pip
pip install gradio diffusers transformers
```

You might see complaints about jax missing for some packages, this is fine we are using pytorch instead and not the modules that specify jax requirement. 

## TODO:

- API via quart and Web Frontend .. May use Gradio but want a custom UI.
- Lots of prompt engineering for the text adventure stuff...
- Ground generated images with appended style prompt and negative prompt
- Save / Load
- Speed up / Optimize
- UI iteration...
- More config...
- Audio?? Music / Character Speech
- Animation ??