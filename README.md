# SAFMN
Code for "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"

### Quantitative results 
  - Benchmark results 

| Degradation | Scale | Params [K] | FLOPs [G] | Acts. [M] | Model Zoo| Visual Results| 
| :----- | :-----: | :-----: | :-----: |:-----: |:-----: |:-----: |
| BI | x2 | 227.82 | 51.53 | 299.00| [Google]()/[Baidu]() Drive | [Google]()/[Baidu]() Drive |
| BI | x3 | 232.69 | 23.42 | 134.00| [Google]()/[Baidu]() Drive |  |
| BI | x4 | 239.52 | 13.56 | 76.70 | [Google]()/[Baidu]() Drive |  |
| BI | x4 (Large) | 5600.82 | 321.00 | 521.00 | [Google]()/[Baidu]() Drive |  |
| [High-order](https://github.com/xinntao/Real-ESRGAN) | x4 (Large)| 5600.82 | 321.00 | 521.00 | [Google]()/[Baidu]() Drive |  |

- Runtime comparison (1080P --> 4K)

| Method | Params [K] | FLOPs [G] | GPU Mem. [M] | Running Time [s]|
| :----- | :-----: | :-----: | :-----: |:-----: |
| IMDN | 715.18 | 1474.41| 7117.48 | 0.26 |
| RLFN | 543.74 | 1075.69| 4973.25 | 0.22 |
| SAFMN| 239.52 | 487.53 | 2304.03 | 0.30 |

