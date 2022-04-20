# APRNet

The pytorch implementation of *"APRNet: A 3D Anisotropic Pyramidal Reversible Network With Multi-Modal Cross-Dimension Attention for Brain Tissue Segmentation in MR Images"* on the iSeg-2017 dataset.




## Requirements

Experiments were performed on an Ubuntu 18.04 workstation with two 24G NVIDIA GeForce RTX 3090 GPUs , CUDA 11.1, and python requirements are:

```
elasticdeform==0.4.9
imgaug==0.2.6
matplotlib==3.3.2
nibabel==3.1.1
opencv-python==4.4.0.42
pandas==1.0.4
Pillow==8.4.0
PyYAML==6.0
scikit-image==0.17.2
scikit-learn==0.23.2
scipy==1.4.1
seaborn==0.11.0
SimpleITK==1.2.4
torch==1.8.0
torchvision==0.9.0
tqdm==4.46.1
numpy==1.19.5
```

If you use the Ranger optimizer, please see:

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer



## Implementation

### Data download

Download the [iSeg-2017](https://iseg2017.web.unc.edu/) dataset after the organizer allows and change data config:

```
./config/data_config/iseg2017_cross_validation_settings.yaml
```



### Data preprocess

Raw Data splitting by k-fold cross-validation:

```
python step1_split_dataset_by_CV.py
```

Data normalization and whole brain extraction:

```
python step2_preprocess_data.py
```

Getting Training and Validation Data：

```
python step3_get_train_and_val_feature_csv.py
```



### Training and Validation

Change model configs:

```
config/model_config/Fold0/APRNet_Fold0.yaml
```

Run：

```
python step4_train_APRNet_Fold0.py
```

### 

## Citation

If our projects are beneficial for your works, please cite:

```
@ARTICLE{9470936,  
author={Zhuang, Yuzhou and Liu, Hong and Song, Enmin and Ma, Guangzhi and Xu, Xiangyang and Hung, Chih-Cheng},  
journal={IEEE Journal of Biomedical and Health Informatics},   
title={APRNet: A 3D Anisotropic Pyramidal Reversible Network With Multi-Modal Cross-Dimension Attention for Brain Tissue Segmentation in MR Images},   
year={2022},
volume={26},
number={2},
pages={749-761},  doi={10.1109/JBHI.2021.3093932}}
```



## Acknowledge

1. [PartiallyReversibleUnet](https://github.com/RobinBruegger/PartiallyReversibleUnet)
2. [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
3. [MRBrainS13](https://mrbrains13.isi.uu.nl/data/) 
4. [iSeg2017](https://iseg2017.web.unc.edu/)

