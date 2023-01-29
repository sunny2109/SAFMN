# SAFMN
Code for "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"

### Quantitative results 
  - Benchmark results 

| Degradation | Scale | Params | FLOPs | Acts. | Model Zoo| Visual Results| 
| :----- | :-----: | :-----: | :-----: |:-----: |:-----: |:-----: |
| BI | x2 | 227.82[K] | 51.53[G] | 299.00[M]| [Google]()/[Baidu]() Drive | [Google]()/[Baidu]() Drive |
| BI | x3 | 232.69[K] | 23.42[G] | 134.00[M]|  |  |
| BI | x4 | 239.52[K] | 13.56[G] | 76.70[M] |  |  |
| BI | x2 (Large)| 5.56[M] | 1,274.00[G] | 2,076.00[M]| [Google]()/[Baidu]() Drive | [Google]()/[Baidu]() Drive |
| BI | x3 (Large)| 5.58[M] | 364.00[G] | 592.00[M]|  |  |
| BI | x4 (Large)| 5.60[M] | 321.00[G] | 521.00[M] |  |  |
| [High-order](https://github.com/xinntao/Real-ESRGAN) | x4 (Large)| 5.60[M] | 321.00[G] | 521.00[M] | [Google]()/[Baidu]() Drive |  |

<img src="./figs/classic_SR.png"/> 

- Runtime comparison (1080P --> 4K)

| Method | Params [K] | FLOPs [G] | GPU Mem. [M] | Running Time [s]|
| :----- | :-----: | :-----: | :-----: |:-----: |
| IMDN | 715.18 | 1474.41| 7117.48 | 0.26 |
| RLFN | 543.74 | 1075.69| 4973.25 | 0.22 |
| SAFMN| 239.52 | 487.53 | 2304.03 | 0.30 |

