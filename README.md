## EPSANet：An Efficient Pyramid Split Attention Block on Convolutional Neural Network

[![Travis](https://img.shields.io/badge/language-Python-red.svg)]()

[![GitHub stars](https://img.shields.io/github/stars/murufeng/EPSANet.svg?style=social&label=Stars)](https://github.com/murufeng/EPSANet)
[![GitHub forks](https://img.shields.io/github/forks/murufeng/EPSANet.svg?style=social&label=Forks)](https://github.com/murufeng/EPSANet)


This repo contains the official Pytorch implementaion code and configuration files of [EPSANet：An Efficient Pyramid Split Attention Block on Convolutional Neural Network](https://arxiv.org/abs/2105.14447).


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
## Training
To train models on ImageNet with 8 gpus run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py -a epsanet50 --data /path/to/imagenet 
```

## Model Zoo

Models are trained with 8 GPUs on both ImageNet and MS-COCO 2017 dataset. 

### Image Classification on ImageNet

|         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) | 
|:---------------------:|:---------:|:--------:|:---------:|:---------:|
| EPSANet-50(Small)             |  22.56     | 3.62     | 77.49 | 93.54 |
| EPSANet-50(Large)             | 27.90     | 4.72    | 78.64 | 94.18 | 
| EPSANet-101(Small)             | 38.90   | 6.82     | 78.43 | 94.11 | 
| EPSANet-101(Large)            | 49.59     | 8.97    | 79.38 | 94.58  |


### Object Detection on MS-COCO 2017

#### Faster R-CNN
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP  | AP_50  |  AP_75| 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :--------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 38.56 | 197.07 | 39.2 | 60.3 | 42.3 | 
|    EPSANet-50(large)  | pytorch |   1x    | 43.85 | 219.64 | 40.9 | 62.1 | 44.6 | 


#### Mask R-CNN
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | AP_50  |  AP_75  | 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 41.20 | 248.53 | 40.0 | 60.9 | 43.3 | 
|    EPSANet-50(large)  | pytorch |   1x    | 46.50 | 271.10 | 41.4 | 62.3 | 45.3 | 

#### RetinaNet
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | AP_50  |  AP_75  | 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 34.78 | 229.32 | 38.2  | 58.1 | 40.6 | 
|    EPSANet-50(large)  | pytorch |   1x    | 40.07 | 251.89 | 39.6  | 59.4 | 42.3 | 


### Instance segmentation with Mask R-CNN on MS-COCO 2017
|model |Params(M) | FLOPs(G) | AP | AP_50 | AP_75 | 
| :----:| :-----: | :-----: |:---------:|:---------:|:---------:|
|EPSANet-50(small) | 41.20 | 248.53 | 35.9 | 57.7 | 38.1 | 
|EPSANet-50(Large) | 46.50 | 271.10 | 37.1 | 59.0 | 39.5 | 

