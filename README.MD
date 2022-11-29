# EdgeCompress: Coupling Multi-Dimenional Model Compression and Dynamic Inference for EdgeAI

This repository is provided for reproducing the experimental results reported in the above paper submitted to **IEEE Transactions on Computer Aided Design of Integrated Circuits and Systems (TCAD)**.

## Overview of The Proposed Framework

### 1. The overall framework
![Framework](/images/framework.png)

### 2. The dynamic inference framework

![dynamic](/images/dynamic.png)

## Environment

* Python: 3.7.10
* Pytorch: 1.7.0
* Numpy: 1.20.2
* CUDA: 11.2
* Yacs: 0.1.8
* Pillow: 8.2.0
* Hardware: RTX-3090

## Preparing Your Data

To reproduce the results reported in the paper, you need to get the datasets ready first.

**ImageNet-1K:**

* Official Introduction 
    >ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use.

1. Download the ImageNet-1K validation set via this [LINK](https://image-net.org/download.php)

2. Extract images using the following command:
   ```
   mkdir val && tar -xvf ILSVRC2012_img_val.tar -C ./val
   ```
   
3. Copy prepare_validation.sh to the val directory and re-organize the validation images with:
   ```
   bash prepare_validation.sh
   ```
   
4. If everything goes right, your dataset should be organized as follows:
   ```
   -- dataset 
   -- val
      --n01440764
         --ILSVRC2012_val_00000293.JPEG
         --ILSVRC2012_val_00002138.JPEG
         --...
      --n01443537
      --...
   ```


## Pretrained Models

### 1. ImageNet

Here we provide the pretrained model weights of our approach for reproducing the experimental results reported in the paper.

| Model | MACs (G) | Top-1 Acc. (%) | Configuration | Weights |
| ------ | ------ | ------ | ------ | ------ |
| Resnet50 (Baseline) | 4.1 | 76.0 | `configs/resnet50.yaml` | [Pretrained](https://www.google.com)
| EC-Static-1 | 1.2 | 73.9 | `configs/ec-static-1.yaml` | [Pretrained](https://www.google.com)
| EC-Static-2 | 1.3 | 74.2 | `configs/ec-static-2.yaml` | [Pretrained](https://www.google.com)
| EC-Static-3 | 1.8 | 75.6 | `configs/ec-static-3.yaml` | [Pretrained](https://www.google.com)
| EC-Static-4 | 2.4 | 76.3 | `configs/ec-static-4.yaml` | [Pretrained](https://www.google.com)
| EC-Static-5 | 3.2 | 76.8 | `configs/ec-static-5.yaml` | [Pretrained](https://www.google.com)
| EC-Static-6 | 4.2 | 77.2 | `configs/ec-static-6.yaml` | [Pretrained](https://www.google.com)



