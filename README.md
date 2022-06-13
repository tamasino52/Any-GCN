# Any-GCN
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftamasino52%2FAny-GCN&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=true"/></a>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>

This repository provides pytorch API implementations existing 3D human pose estimation models using GCN.
Models implemented in the API are provided with a standardard residual architecture and use standard hyperparameters different from those used in existing papers, so there may be differences in the performance proposed in the paper. The references of GCN blocks provided by this repository are as follows.


[A Comprehensive Study of Weight Sharing in Graph Networks for 3D Human Pose Estimation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550324.pdf) by Kenkun Liu, Rongqi Ding, Zhiming Zou, Le Wang, Wei Tang.

```
@inproceedings{10.1007/978-3-030-58607-2_19,
  author = {Liu, Kenkun and Ding, Rongqi and Zou, Zhiming and Wang, Le and Tang, Wei},
  title = {A Comprehensive Study of Weight Sharing in Graph Networks for 3D Human Pose Estimation},
  year = {2020},
  isbn = {978-3-030-58606-5},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-030-58607-2_19},
  doi = {10.1007/978-3-030-58607-2_19},
  booktitle = {Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X},
  pages = {318–334},
  numpages = {17},
  location = {Glasgow, United Kingdom}
}
```

[Modulated Graph Convolutional Network for 3D Human Pose Estimation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zou_Modulated_Graph_Convolutional_Network_for_3D_Human_Pose_Estimation_ICCV_2021_paper.pdf) by Zhiming Zou and Wei Tang.

```
@InProceedings{Zou_2021_ICCV,
  author    = {Zou, Zhiming and Tang, Wei},
  title     = {Modulated Graph Convolutional Network for 3D Human Pose Estimation},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {11477-11487}
}
```

## Introduction

We evaluate models for 3D human pose estimation on the [Human3.6M Dataset](http://vision.imar.ro/human3.6m/).

In this repository, only 2D joints of the human pose are exploited as inputs. We utilize the method described in Pavllo et al. [2] to normalize 2D and 3D poses in the dataset. To be specific, 2D poses are scaled according to the image resolution and normalized to [-1, 1]; 3D poses are aligned with respect to the root joint. Please refer to the corresponding part in Pavllo et al. [2] for more details. For the 2D ground truth, we predict 16 joints (as the skeleton in Martinez et al. [1] and Zhao et al. [3] without the 'Neck/Nose' joint). For the 2D pose detections, the 'Neck/Nose' joint is reserved. 


## Quickstart

This repository is build upon Python v3.8 and Pytorch v1.9.0 on Windows 11. All experiments are conducted on a single NVIDIA RTX 3090 GPU. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v3.78 from [Anaconda](https://www.anaconda.com/) and installing Pytorch (>= 1.9.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone https://github.com/tamasino52/Any-GCN.git
cd Any_GCN
pip install -r requirements.txt
```

### Benchmark setup
CPN 2D detections for Human3.6M datasets are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) Pavllo et al. [2], which can be downloaded by the following steps:

```
cd dataset
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
```

3D labels and ground truth can be downloaded
```
cd dataset
gdown --id 1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0
cd ..
```

### GT setup 

GT 2D keypoints for Human3.6M datasets are obtained from [SemGCN](https://github.com/garyzhao/SemGCN) Zhao et al. [3], which can be downloaded by the following steps:
```
cd data
pip install gdown
gdown https://drive.google.com/uc?id=1Ac-gUXAg-6UiwThJVaw6yw2151Bot3L1
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```
After this step, you should end up with two files in the dataset directory: data_3d_h36m.npz for the 3D poses, and data_2d_h36m_gt.npz for the ground-truth 2D poses.

### GT Evaluation 
```
python main_gcn.py --model {model_name}  --eva checkpoint/{weight_name}.pth.tar
```

### GT Training 
```
# Vanilla GCN
python main_gcn.py --model vanilla

# Decoupled Vanilla GCN
python main_gcn.py --model dc_vanilla

# Pre-Aggresive GCN
python main_gcn.py --model preagg

# Decoupled Pre-Aggresive GCN
python main_gcn.py --model dc_preagg

# Post-Aggresive GCN
python main_gcn.py --model postagg

# Decoupled Post-Aggresive GCN
python main_gcn.py --model dc_postagg

# Convolution-style GCN
python main_gcn.py --model convst

# No-sharing GCN
python main_gcn.py --model nosharing

# Modulated GCN
python main_gcn.py --model modulated
```

### Acknowledgement
This code is extended from the following repositories.
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [Semantic GCN](https://github.com/garyzhao/SemGCN)
- [Local-to-Global GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [Modulated-GCN](https://github.com/ZhimingZo/Modulated-GCN)
We thank the authors for releasing their code. Please also consider citing their work.
