## IOPE-Net: The Water IOPs Estimation Network

Generating accurate estimation of water inherent optical properties (IOPs) from hyperspectral images plays a significant role in marine exploration. Traditional methods mainly adopt bathymetric models and numerical optimization algorithms to deal with this problem. However, these methods usually tend to simplify the bathymetric models and lack the capability of describing the discrepancy between reference spectrum and estimation spectrum, resulting in a limited estimation performance. To get a more precise result, in this work, we propose a novel network based on deep learning to retrieve the IOPs. The proposed network, named as IOPs estimation network (IOPE-Net), explores a hybrid sequence structure to establish IOPs estimation module that acquires high-dimensional nonlinear features of water body spectrums for water IOPs estimation. Moreover, considering the insufficiency of labeled training samples, we design a spectrum reconstruction module combined with classical bathymetric model to train the proposed network in an unsupervised manner. Experimental results on both simulated and real datasets demonstrate the effectiveness and efficiency of our method in comparison with the state-of-the-art water IOPs retrieving methods.
* **Rreference:** [2021, Qi, Hybrid Sequence Networks for Unsupervised Water Properties Estimation From Hyperspectral Imagery]

* **Input:** Hyperspectral Imagery(HSI) of ***optically deep water***
* **Output:** Inherent Optical Properties(IOPs) of the water, namely, ***absorption coefficient (a)*** , ***backward scattering coefficient (bb)***

<div align=center><img src="https://github.com/lijinchao98/IOPE_Net/blob/main/fig.jpg" width="600px" alt="IOPE-Net"></div>

The first part contains a IOPs estimation module, which is used for estimating absorption rate a(λ) and scattering rate bb(λ) from the input hyperspectral imagine concurrently. Consequently, the outputs of this part are the desired IOPs estimation results. The second part is designed for unsupervised training methodology. In this part, a spectrum reconstruction module has been established based on the classic bathymetric model, and we employ it to reconstruct sensor-observed spectrum with above IOPs estimation results. At the same time, the reconstruction errors evaluated subsequently are utilized to adjust the weight parameters of IOPs estimation module.

(https://doi.org/10.1109/JSTARS.2021.3068727). 
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
