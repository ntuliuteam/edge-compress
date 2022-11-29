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

**CIFAR-10**

* Official Introduction
    >  The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

1. Download CIFAR-10 dataset via this [LINK](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Extract images with the following command:
   ```
   tar -zxvf cifar-10-python.tar.gz
   ```

3. Finally, the dataset folder should have the following architecture:
   ```
   --cifar-10-batches-py
   --batches.meta
   --readme.html
   --data_batch_1
   --data_batch_2
   --data_batch_3
   --data_batch_4
   --data_batch_5
   --test_batch
   ```
