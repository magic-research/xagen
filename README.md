<!-- # showlab.github.io/xagen -->

<p align="center">

  <h2 align="center">XAGen: 3D Expressive Human Avatars Generation</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=-4iADzMAAAAJ&hl=en"><strong>Zhongcong Xu</strong></a>
    ·
    <a href="http://jeff95.me/"><strong>Jianfeng Zhang</strong></a>
    ·
    <a href="https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en"><strong>Jun Hao Liew</strong></a>
    ·
    <a href="https://sites.google.com/site/jshfeng/home"><strong>Jiashi Feng</strong></a>
    ·
    <a href="https://sites.google.com/view/showlab"><strong>Mike Zheng Shou</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2311.13574">
        <img src='https://img.shields.io/badge/arXiv-XAGen-red' alt='Paper PDF'>
        </a>
        <a href='https://showlab.github.io/xagen'>
        <img src='https://img.shields.io/badge/Project_Page-XAGen-green' alt='Project Page'>
        </a>
  </p>
  
  <table align="center">
    <td>
      <p align="center">
        <img src="assets/Teaser.png" width="500">
      </p>
    </td>
  </table>

## ⚒️ Installation
prerequisites: `python>=3.7`, `CUDA>=11.3`.

Install with `conda` activated: 
```bash
source ./install_env.sh
```

Follow the instructions in this [repo](https://github.com/yfeng95/PIXIE/blob/master/Doc/docs/getting_started.md#requirements) and [website](https://expose.is.tue.mpg.de/) to download parametric models and place the parametric models as follow:
```bash
xagen
|----smplx
  |----assets
    |----MANO_SMPLX_vertex_ids.pkl
    |----SMPL-X__FLAME_vertex_ids.npy
    |----smplx_canonical_body_sdf.pkl
    |----smplx_extra_joints.yaml
    |----SMPLX_NEUTRAL_2020.npz
    |----SMPLX_to_J14.pkl
```

## 🏃‍♂️ Getting Started
Due to the copyright issue, we are unable to release all the processed datasets, we provide a sampled dataset and all the dataset labels for inference. Please download the sampled datasets and pretrained checkpoints from [release](https://github.com/magic-research/xagen/releases/tag/public_release). Then modify the path to data and checkpoints in the scripts.
Run training:
```bash
bash dist_train.sh
```
Run inference:
```bash
bash inference.sh
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{XAGen2023,
    title={XAGen: 3D Expressive Human Avatars Generation},
    author={Xu, Zhongcong and Zhang, Jianfeng and Liew, Junhao and Feng, Jiashi and Shou, Mike Zheng},
    booktitle={NeurIPS},
    year={2023}
}
```

