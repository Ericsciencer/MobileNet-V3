# MobileNetV3
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result

<img width="941" height="335" alt="image" src="https://github.com/user-attachments/assets/200ed517-98f1-4229-a74d-c38cd27d3fa7" />
<img width="2480" height="1914" alt="mobilenetv3_training_curve" src="https://github.com/user-attachments/assets/d5627c56-e16b-4019-bd6b-f71aef30d1db" />


---

## 简介
MobileNetV3 是由谷歌团队于 2019 年提出的**新一代轻量化深度卷积神经网络**，相关成果发表于《Searching for MobileNetV3》。在 MobileNetV1 深度可分离卷积、MobileNetV2 倒残差线性瓶颈结构的基础上，针对移动端硬件延迟、算力消耗、特征表达能力做全方位升级，引入**神经架构搜索NAS + NetAdapt 双阶段优化策略**，解决人工设计轻量网络难以兼顾精度、速度与硬件适配性的痛点。其核心创新包含：**硬件感知NAS自动搜索最优网络架构**、**Bneck倒残差瓶颈模块**、**HardSwish/HardSigmoid 轻量化激活函数**、**改进版SE通道注意力模块**、**混合3×3/5×5卷积核配置**，同时分为 Large、Small 两个版本适配不同性能设备。模型在极低参数量与计算量下，进一步拉高分类精度，推理延迟显著低于 MobileNetV1/V2，完美适配手机、嵌入式、无人机、边缘计算等终端设备，成为工业界轻量化模型部署的主流基准骨架，广泛应用于图像分类、目标检测、语义分割、人脸识别等计算机视觉任务。

## 架构
MobileNetV3 整体为**NAS搜索优化的倒残差堆叠轻量化卷积网络**，整体分为「Stem初始卷积模块」「Bneck倒残差特征提取模块」和「尾部卷积+全局池化+全连接分类模块」三大核心部分，原论文标准输入为224×224分辨率的3通道RGB图像，提供 Large/Small 两种结构规格适配不同场景，具体核心设计如下：
- **Stem初始基础模块**：网络首层采用标准3×3卷积，搭配BN批量归一化与HardSwish激活函数，替代传统ReLU，在降低计算量的同时增强浅层非线性特征表达，完成图像基础特征提取与初始下采样。
- **轻量化特征提取模块（核心）**：全程堆叠**Bneck倒残差瓶颈模块**，由NAS神经架构搜索自动确定每一层的扩展通道数、卷积核尺寸(3×3/5×5)、步长、是否接入SE注意力、选用ReLU/HardSwish激活；模块内部遵循「1×1升维卷积 → 深度卷积 → 轻量化SE注意力(可选) → 1×1降维卷积」结构，加入残差旁路连接，在控制计算量的同时强化特征复用与通道建模能力。
- **分类输出模块**：后端增设1×1升维卷积整合高层语义特征，采用全局平均池化压缩特征图维度，减少冗余参数量；末端单层全连接层映射分类维度，精简分类头结构，兼顾推理速度与分类精度。

该架构首次将**硬件感知NAS**融入轻量网络设计，不再依赖人工调参，自动搜索适配移动端CPU的最优层配置、模块组合与超参数，结合倒残差、轻量化激活、改进SE注意力多重优化，实现了**更小参数量、更低延迟、更高精度**的三重突破，是现阶段移动端、边缘端视觉任务最优基础骨架之一。

增加SE机制的Bottleneck模块结构：
<img width="739" height="280" alt="image" src="https://github.com/user-attachments/assets/0a9ab858-fbb3-49cc-9aa3-e9efc3c22d60" />
h-swish替代RELU6激活函数：
<img width="336" height="85" alt="image" src="https://github.com/user-attachments/assets/f084eb90-0a00-4e59-9874-4a97cb4ccc0e" />
<img width="830" height="355" alt="image" src="https://github.com/user-attachments/assets/c41195fb-1959-49ee-a1f8-3ca6749d282e" />
<img width="750" height="373" alt="image" src="https://github.com/user-attachments/assets/3a334d8e-a0eb-4248-ad65-c396707f13c4" />

优化输出Stage结构：
<img width="1040" height="440" alt="image" src="https://github.com/user-attachments/assets/5397f1e7-4d5b-444b-aecd-c882c1d26440" />

整体网络（分大与小）：
<img width="631" height="630" alt="image" src="https://github.com/user-attachments/assets/ccf03974-3b12-4fc7-87e7-2a64a2fd8b98" />
<img width="631" height="471" alt="image" src="https://github.com/user-attachments/assets/61fa3a02-6ae7-4f92-a7f6-7a70dd9a9aa7" />

**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要调整初始卷积与部分模块下采样步长、防止特征图尺寸过小），但核心架构**Bneck倒残差模块 + NAS搜索拓扑 + HardSwish激活 + 轻量化SE注意力**完全保留，严格复现原版MobileNetV3核心设计思想。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

---

## Introduction
MobileNetV3 is a new lightweight deep convolutional neural network proposed by the Google team in 2019, published in the paper *Searching for MobileNetV3*. On the basis of MobileNetV1’s depthwise separable convolution and MobileNetV2’s inverted residual linear bottleneck, it comprehensively optimizes mobile hardware latency, computational consumption and feature representation capability. It adopts a **two-stage optimization strategy of hardware-aware NAS + NetAdapt**, solving the pain point that manually designed lightweight networks cannot balance accuracy, speed and hardware adaptability. Its core innovations include: hardware-aware NAS for automatically searching the optimal network architecture, Bneck inverted residual bottleneck module, HardSwish/HardSigmoid lightweight activation function, improved SE channel attention module, and hybrid 3×3/5×5 convolution kernel configuration. It is divided into Large and Small versions to adapt to devices with different performance. With extremely low parameters and computational cost, the model further improves classification accuracy, and its inference latency is significantly lower than MobileNetV1/V2. It is widely deployed in mobile phones, embedded devices, UAVs, edge computing and other terminal scenarios, and has become a mainstream baseline backbone for industrial lightweight models, applied to image classification, object detection, semantic segmentation, face recognition and other visual tasks.

## Architecture
The overall structure of MobileNetV3 is a lightweight convolutional network with inverted residual stacking optimized by NAS. It is divided into three core parts: the Stem initial convolution module, the Bneck inverted residual feature extraction module, and the tail convolution + global pooling + fully connected classification module. The original paper adopts 224×224 RGB images as standard input, and provides two specifications of Large/Small to adapt to different application scenarios.
- **Stem Initial Basic Module**: The first layer uses standard 3×3 convolution with BN batch normalization and HardSwish activation function instead of traditional ReLU. It enhances shallow nonlinear feature expression while reducing computational cost, completing initial shallow feature extraction and downsampling.
- **Lightweight Feature Extraction Module (Core)**: Stacked with Bneck inverted residual bottleneck modules. The hardware-aware NAS automatically determines the expansion channel, convolution kernel size (3×3/5×5), stride, whether to use SE attention, and activation type (ReLU/HardSwish) for each layer. The internal structure follows the order of 1×1 expansion convolution → depthwise convolution → optional lightweight SE attention → 1×1 compression convolution, with residual shortcut connection to strengthen feature reuse and channel modeling capability.
- **Classification Output Module**: The tail adds 1×1 convolution to integrate high-level semantic features. Global average pooling is used to compress feature map dimensions and reduce redundant parameters. The final fully connected layer maps to classification dimensions with a streamlined classification head, balancing inference speed and classification accuracy.


Bottleneck module structure with added SE mechanism:
<img width="623" height="730" alt="image" src="https://github.com/user-attachments/assets/36b02a14-089e-4003-af80-13265cdf0fc3" />

Optimize the output Stage structure:
<img width="1040" height="440" alt="image" src="https://github.com/user-attachments/assets/5397f1e7-4d5b-444b-aecd-c882c1d26440" />

Non-linear startup function:
<img width="1050" height="289" alt="image" src="https://github.com/user-attachments/assets/f2e90420-b67f-456c-9efe-3d051bafdb40" />


Overall network (divided into large and small):
<img width="631" height="630" alt="image" src="https://github.com/user-attachments/assets/ccf03974-3b12-4fc7-87e7-2a64a2fd8b98" />
<img width="631" height="471" alt="image" src="https://github.com/user-attachments/assets/61fa3a02-6ae7-4f92-a7f6-7a70dd9a9aa7" />

**Note:** We use the CIFAR-10 dataset with 10 classification categories. Since the 32×32 image size of CIFAR-10 is much smaller than the 224×224 input in the original paper, slight adjustments are made to the downsampling stride of the initial convolution and partial modules to avoid excessive feature compression. However, the core designs of **Bneck inverted residual module, NAS searched topology, HardSwish activation and lightweight SE attention** are completely consistent with the original MobileNetV3.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

---
## 原文章 | Original article
Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1314-1324.
