# Semantic Indoor Place Recognition

Repo will be cleaned up soon.

![Overview](./doc/overview.png)

## Introduction

This repository contains the implementation of **CGiS-Net** in [PyTorch](https://pytorch.org/).
 
CGiS-Net is an indoor place recognitino network presented in our IROS 2022 paper ([arXiv](https://arxiv.org/abs/2202.02070)). If you find our work useful in your research, please consider citing:

```
@inproceedings{ming2022CGiSNet,
    author = {Ming, Yuhang and Yang, Xingrui and Zhang, Guofeng and Calway, Andrew},
    title = {CGiS-Net: Aggregating Colour, Geometry and Implicit Semantic Features for Indoor Place Recognition}
    booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    Year = {2022}
}
```

## Installation

This implementation has been tested on Ubuntu 18.04 and 20.04. 

* For Ubuntu 18.04 installation, please see the instructions from the official KP-Conv repository [INSTALL.md](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md).

* For Ubuntu 20.04 installation, the procedure is basically the same except for different versions of packages are used.

    - PyTorch 1.8.0, torchvision 0.9.0, CUDA 11.1, cuDNN 8.6.0

## Experiments

### Generate data


### Training stage 1:


### Training stage 2:


### Visualisations
* [Kernel Visualization](./doc/visualization_guide.md): Use the script from KP-Conv repository, the kernel deformations can be displayed.

### Results

Our CGiS-Net is compared to a traditional baseline using SIFT+BoW, and 4 deep learning based method [NetVLAD](https://github.com/Nanne/pytorch-NetVlad), [PointNetVLAD](https://github.com/cattaneod/PointNetVlad-Pytorch), [MinkLoc3D](https://github.com/jac99/MinkLoc3D) and Indoor DH3D.

|   |Recall@1|Recall@2|Recall@3|
|---|---|---|---|
| SIFT+BoW  | 16.16  | 21.17  | 24.38  |
| NetVLAD  | 21.77  | 33.81  | 41.49  |
| PointNetVLAD  | 5.31  | 7.50  | 9.99  |
| MinkLoc3D  | 3.32  | 5.81  | 8.27  |
| Indoor DH3D  | 16.10  | 21.92  | 25.30  |
| CGiS-Net (Ours)  | 61.12  | 70.23  | 75.06  |

![Results](./doc/results.png)


## Acknowledgment

In this project, we use parts of the official implementations of following works:

* <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KP-FCNN</a> (Semantic Encoder-Decoder)

* <a href="https://github.com/cattaneod/PointNetVlad-Pytorch">PointNetVLAD-Pytorch</a> (NetVLAD Layer)

