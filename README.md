## EPSANet：An Efficient Pyramid Split Attention Block on Convolutional Neural Network

This repo contains the Pytorch implementaion code and configuration files of [EPSANet：An Efficient Pyramid Split Attention Block on Convolutional Neural Network](https://arxiv.org/abs/2105.14447). created by Hu Zhang.

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.0+

### Our environments

- OS: Ubuntu 18.04
- CUDA: 10.0
- Toolkit: PyTorch 1.0
- GPU: Titan RTX

## Data preparation

Download and extract ImageNet train and val images from [http://image-net.org/](http://image-net.org/).
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

# Usage
First, clone the repository locally:
```
git clone https://github.com/murufeng/EPSANet.git
cd EPSANet
```
- Create a conda virtual environment and activate it:

```bash
conda create -n epsanet python=3.6 
conda activate epsanet
```

- Install `CUDA==10.0` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.0.1` and `torchvision==0.2.0` with `CUDA==10.0`:

```
conda install -c pytorch pytorch torchvision
```
