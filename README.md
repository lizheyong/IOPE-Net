## IOPE_Net
* The  **Hyperspectral Image(HSI**) of ***optical deep water*** is input, using the IOPE_Net in the form of an autoencoder, the Inherent Optical Properties(IOPs) of the water, namely, the **absorption coefficient (a)** and the **backward scattering coefficient (bb)** are estimated unsupervised . 
* The code is written with reference to 

[1] [A Self-Improving Framework for Joint Depth Estimation and Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs13091721). 

* The code implementation may have some imperfections, please feel free to submit issues.
* The article proposes a hybrid sequence CNN+RNN, but in my implementation, CNN alone is sufficient and more mainly faster than adding RNN, so I only use CNN in my subsequent work.

## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
