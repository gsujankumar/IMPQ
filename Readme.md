# IMPQ : Reduced Complexity Neural Networks via Granular Precision Assignment

## Introduction

This repo contains the PyTorch implementation of for the paper *IMPQ : Reduced Complexity Neural Networks via Granular Precision Assignment*, which will appear in in ICASSP 2022.  

### Abstract

The demand for the deployment of deep neural networks (DNN) on resource-constrained Edge platforms is ever increasing. Today's DNN accelerators support mixed-precision computations to enable reduction of computational and storage costs but require networks with precision at variable granularity (network vs. layer vs. kernel level). However, the problem of granular precision assignment is challenging due to an exponentially large search space and efficient methods for such precision assignment are lacking. To address this problem, we introduce the iterative mixed-precision quantization (IMPQ) framework to allocate precision at variable granularity. IMPQ employs a sensitivity metric to order the weight/activation groups in terms of the likelihood of misclassifying input samples due to its quantization noise. It iteratively reduces the precision of the weights and activations of a pretrained full-precision network starting with the least sensitive group. Compared to state-of-the-art methods, IMPQ reduces computational costs by 2x-to-2.5x for compact networks such as MobileNet-V1 on ImageNet with no accuracy loss. Our experiments reveal that kernel-wise granular precision assignment provides 1.7x higher compression than layer-wise assignment.


## Dependencies

We evaluate this code with PyTorch 1.6 on Python 3.7. We require standard packages such as `torchvision`, `numpy`, `scikit-learn` and `cuda`.

You will need to download ImageNet dataset. And change the path to dataset appropriately.

### CIFAR10 experiments

##### To run the code to generate CIFAR10 on ResNet20 floating point baseline network use this:

```
python main.py --arch ResNet20 --quantize Dont
 --dataset CIFAR10 --log-dir <dirname> --lr-scheduler step
 --epochs 200
```

##### Running IMPQ

* First generate CIFAR10 on ResNet20 network with 4-bit precision :

```
python main.py --arch ResNet20 --quantize Lin
 --dataset CIFAR10 --log-dir <dirname>  --lr-scheduler step
 --epochs 200 --nbits 4 
```

* Next, generate reduced bit precision networks using IMPQ use this:

```
python main.py --arch ResNet20 --quantize Lin --dataset CIFAR10 
--log-dir <dirname> --mode reduce --lr-scheduler step
--checkpoint <path to starting network>
```

### ImageNet experiments

##### To run the code to generate ImageNet on MobileNet-V1 floating point baseline network use this:

```
python main.py --arch MobileNet  --quantize Dont 
--log-dir <dirname> --dataset ImageNet
```

##### Running IMPQ

* First generate ImageNet on MobileNet-V1 network with 4-bit precision :

```
python main.py --arch MobileNet  --quantize Lin 
--log-dir <dirname> --dataset ImageNet --nbits 8 
```

* Next, generate reduced bit precision with weights only quantization using IMPQ:

```
python main.py --arch MobileNet  --quantize Lin 
--log-dir <dirname> --mode reduce --dataset ImageNet
--checkpoint <path to starting network>
```


* Next, generate reduced bit precision with weights and activation quantization using IMPQ:

```
python main.py --arch MobileNet  --quantize Lin 
--log-dir <dirname> --mode actquant --dataset ImageNet
--checkpoint <path to starting network>
```

### How to cite this paper

Use the following `bibtex` entry to cite this paper:

```
@inproceedings{gonu2022IMPQ,
  title={IMPQ : Reduced Complexity Neural Networks via Granular Precision Assignment},
  author={Gonugondla, Sujan Kumar and Shanbhag, Naresh},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
}
```
