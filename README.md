## IOPE_Net
* **Input:** Hyperspectral Imagery(HSI) of ***optically deep water***
* **Output:** Inherent Optical Properties(IOPs) of the water, namely, ***absorption coefficient (a)*** , ***backward scattering coefficient (bb)***
* **Network:** Autoencoder, model driven + data driven, estimated unsupervised 

<div align=center><img src="https://github.com/lijinchao98/IOPE_Net/blob/main/fig.jpg" width="600px" alt="IOPE-Net"></div>

* Code is written with reference to [2021, Jiaohao Qi, A Self-Improving Framework for Joint Depth Estimation and Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs13091721). 
* The article proposes a hybrid sequence CNN+RNN, but in my implementation, CNN alone is sufficient and more mainly faster than adding RNN, so I only use CNN in my subsequent work.
* For the Loss function, here only use MSE Loss and SA Loss. HMS Loss was not used, yes, because I was too lazy to add it.
* Code implementation may have some imperfections, please feel free to submit issues.*
## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## Data Format
* Hyperspectral data (.hdr) is converted to (.npy) by 'hdr_2_npy' and then flattened to the shape (pixels, wavelength) before use it.
* All data used are reflectance data, i.e. after reflectance correction.
