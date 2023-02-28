# Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution
[Long Sun](https://github.com/sunny2109), [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN), [Jinshan Pan](https://jspan.github.io/)

[IMAG Lab](https://imag-njust.net/), Nanjing University of Science and Technology

---
[![GitHub Stars](https://img.shields.io/github/stars/sunny2109/SAFMN?style=social)](https://github.com/sunny2109/SAFMN)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=sunny2109/SAFMN)  [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2302.13800.pdf)  ![update](https://badges.strrl.dev/updated/sunny2109/SAFMN)


---
<p align="center">
  <img width="800" src="./figs/framework.png">
</p>

*An overview of the proposed SAFMN. SAFMN first transforms the input LR image into the feature space using a convolutional layer, performs feature extraction using a series of feature mixing modules (FMMs), and then reconstructs these extracted features by an upsampler module. The FMM block is mainly implemented by a spatially-adaptive feature modulation (SAFM) layer and a convolutional channel mixer (CCM).*


### Training


### Testing 


### Results
- **Pretrained models and visual results**

| Degradation | Model Zoo| Visual Results| 
| :----- |:-----: |:-----: |
| BI-Efficient SR | [Google Drive](https://drive.google.com/drive/folders/12O_xgwfgc76DsYbiClYnl6ErCDrsi_S9?usp=share_link)/[Baidu Netdisk](https://pan.baidu.com/s/1mKXahFifHaF14pc1pBWFOg) with code: SAFM | [Google Drive](https://drive.google.com/drive/folders/1s3vJQXDACr799khLLs1ELWL-neljx5vL?usp=share_link)/[Baidu Netdisk](https://pan.baidu.com/s/17q_OuNVTgy7QhtbFu099Jg) with code: SAFM |
| BI-Classic SR | [Google Drive](https://drive.google.com/drive/folders/12O_xgwfgc76DsYbiClYnl6ErCDrsi_S9?usp=share_link)/[Baidu Netdisk](https://pan.baidu.com/s/10jtlG-FYfB8KwYfWsQDOMA) with code: SAFM | [Google Drive](https://drive.google.com/drive/folders/1s3vJQXDACr799khLLs1ELWL-neljx5vL?usp=share_link)/[Baidu Netdisk](https://pan.baidu.com/s/1fYsZ67MNLpPs7OAS9Dn2-w) with code: SAFM |
| x4 [High-order](https://github.com/xinntao/Real-ESRGAN) |[Google Drive](https://drive.google.com/drive/folders/12O_xgwfgc76DsYbiClYnl6ErCDrsi_S9?usp=share_link)/[Baidu Netdisk](https://pan.baidu.com/s/10jtlG-FYfB8KwYfWsQDOMA) with code: SAFM |  |

- **Efficient SR Results**
<img src="./figs/efficient_sr.png">

- **Classic SR Results**
<img src="./figs/classic_sr.png">

- **Runtime Comparison**
<img src="./figs/runtime.png">

- **Comparison with NTIRE Winners**
<img width="800" src="./figs/esr_winner.png">


### Citation
    @article{sun2023safmn,
      title={Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution},
      author={Sun, Long and Dong, Jiangxin and Tang, Jinhui and Pan, Jinshan},
      journal={arXiv preprint arXiv:2302.13800},
      year={2023}
    }


### Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

