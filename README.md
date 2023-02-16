## IOPE_Net
* **Hyperspectral Image(HSI**) of ***optically deep water*** is input, using the IOPE_Net in the form of an autoencoder, the Inherent Optical Properties(IOPs) of the water, namely, the **absorption coefficient (a)** and the **backward scattering coefficient (bb)** are estimated unsupervised . 
* Code is written with reference to [2021, Jiaohao Qi, A Self-Improving Framework for Joint Depth Estimation and Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs13091721). 
* The article proposes a hybrid sequence CNN+RNN, but in my implementation, CNN alone is sufficient and more mainly faster than adding RNN, so I only use CNN in my subsequent work.
* Code implementation may have some imperfections, please feel free to submit issues.

## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## Data Format
* Hyperspectral data (.hdr) is converted to (.npy) by 'hdr_2_npy' and then flattened to the shape (pixels, wavelength) before use it.
* All data used are reflectance data, i.e. after reflectance correction.

## 
