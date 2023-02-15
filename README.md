## IOPE_Net 水体固有光学性质估计网络/水质估计/水质反演
The code is written with reference to [A Self-Improving Framework for Joint Depth Estimation and Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs13091721). The code implementation may have some imperfections, please feel free to submit issues.
The article proposes a hybrid sequence CNN+RNN, but in my implementation, CNN alone is sufficient and more mainly faster than adding RNN, so I only use CNN in my subsequent work.

## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
