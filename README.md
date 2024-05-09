# SWIN-TOD
This is the official implementation of the paper "SWIN-TOD: Smooth Wasserstein Distance and Instance-level Neighboring Enhancement for Remote Sensing Tiny Object Detection".

## Introduction
SWIN-TOD contains two main modules: Instance-level Neighboring Enhancement Network (INEN) and the Smooth Wasserstein Loss (SWL).

**Abstract**: The advancement of deep neural network technology has propelled the widespread application of remote sensing target detection. However, compared to natural scenes, remote sensing targets possess inherent characteristics such as weak features and small scales, leading to a significant performance gap in traditional detection methods. To address these challenges, a principled analysis of existing approaches is conducted, focusing on two key aspects: inadequate extraction of discriminative features and inappropriate regression measurement metric. To tackle the first issue, an Instance-level Neighboring Enhancement Network (INEN) is proposed, enhancing the network's feature extraction capability through inter-object feature learning. To address the second issue, a novel metric parameter, Smooth Wasserstein Loss (SWL), is devised. Building upon these principles, a remote sensing small target detection network is developed. Extensive experiments on AI-TOD v1/v2 and DOTA v2 remote sensing tiny target detection datasets demonstrate that our approach achieves state-of-the-art (SOTA) performance.

**Challenges**:
![demo image](figures/Fig1.png)

**Pipeline of network**:
![demo image](figures/Fig3.png)


**INEN**:
![demo image](figures/Fig4.png)


## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install:

Note that this repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/sevenwgb/SWIN-TOD.git
cd SWIN-TOD
pip install -r requirements/build.txt
python setup.py develop
```

## Visualization
The images are from the AI-TOD v1/v2, and DOTA-v2 datasets. Note that the <font color=green>green box</font> denotes the True Positive, the <font color=red>red box</font> denotes the False Negative and the <font color=blue>blue box</font> denotes the False Positive predictions.
**High-quality results**:
![demo image](figures/Fig5-0.png)
**Effectiveness of INEN**:
![demo image](figures/Fig9.png)
**Effectiveness of SWL**:
![demo image](figures/Fig10.png)

## Citation

