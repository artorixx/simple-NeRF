# Simple-NeRF

This is a simple implement of NeRF, as presented in the paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://www.matthewtancik.com/nerf) by Mildenhall et al., ECCV2020.  
The original repository of the paper is ["here"](https://github.com/bmild/nerf) and this repository is based on ["a PyTorch implement of the repository"](https://github.com/yenchenlin/nerf-pytorch).  

## Features
- This source code makes the original source work on multiple GPUs.
- This works with only blender dataset not liff dataset etc.

## Setup

The following steps get you started.

### Installation
```
git clone https://github.com/artorixx/simple-nerf.git
cd simple-nerf
pip install -r requirements.txt
```
### Dataset
Look at a download procedure of the original repository.  
1. you make directory `data`.  
2. you put `lego` (or what you want to train) into `data`. (Like `data/lego/...`)

### Training
Specify the path of a config file at the head of train.py.  
To train a model on a single GPU, simply run e.g.:
```
python train.py
```

To train a model on multiple GPUs on a single machine, launch multiple processes via Torchrun, where $NUM_GPUS is the number of GPUs to use:

```
screen torchrun --standalone --nnodes 1 --nproc_per_node $NUM_GPUS train.py
```

### Visualizing demo
Specify the path of a config file at the head of train.py.  
```
python visualize.py
```
Demo movie is generated in the same directory as a config file.  

## Config
You can control the experiment setting by editing a config file.  
The following terms are arguments for config.  
>`model` Hyperparameters for model  
>`training` Arguments for training
>>`batch_size` How many images processed in an iteration
>`ray_size` How many rays processed in an iteration  
>`peak_lr` Peak learning rate for learning rate scheduler  
>`peak_it` Peak iteration for learning rate scheduler  
>`decay_it` Decay iteration for learning rate scheduler  
>`decay_rate` Decay rate for learning rate scheduler  
>`N_coarse_samples` Number of coarse sampling  
>`N_fine_samples` Number of fine sampling  
>`precrop_iters` How many iterations of precroping (In the initial training, rays in only precrop area are processed)
>`precrop_frac` Precrop area ratio 




