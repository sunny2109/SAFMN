# SAFMN
Code for "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"

### Quantitative results 
  - Benchmark results 

| Degradation | Model Zoo| Visual Results| 
| :----- |:-----: |:-----: |
| BI | Efficient SR: [Google]()/[Baidu]() Drive | [Google]()/[Baidu]() Drive |
| BI | Classic SR: [Google]()/[Baidu]() Drive | [Google]()/[Baidu]() Drive |
| x4 [High-order](https://github.com/xinntao/Real-ESRGAN) |[Google]()/[Baidu]() Drive |  |

<img src="./figs/Efficient_SR.png"/> 

<img src="./figs/classic_SR.png"/> 


- Runtime comparison (1080P --> 4K)

| Method | Params [K] | FLOPs [G] | GPU Mem. [M] | Running Time [s]|
| :----- | :-----: | :-----: | :-----: |:-----: |
| IMDN | 715.18 | 1474.41| 7117.48 | 0.26 |
| RLFN | 543.74 | 1075.69| 4973.25 | 0.22 |
| SAFMN| 239.52 | 487.53 | 2304.03 | 0.30 |

