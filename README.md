## IOPE-Net: The Water IOPs Estimation Network

Generating accurate estimation of water inherent optical properties (IOPs) from hyperspectral images plays a significant role in marine exploration. Traditional methods mainly adopt bathymetric models and numerical optimization algorithms to deal with this problem. However, these methods usually tend to simplify the bathymetric models and lack the capability of describing the discrepancy between reference spectrum and estimation spectrum, resulting in a limited estimation performance. To get a more precise result, in this work, we propose a novel network based on deep learning to retrieve the IOPs. The proposed network, named as IOPs estimation network (IOPE-Net), explores a hybrid sequence structure to establish IOPs estimation module that acquires high-dimensional nonlinear features of water body spectrums for water IOPs estimation. Moreover, considering the insufficiency of labeled training samples, we design a spectrum reconstruction module combined with classical bathymetric model to train the proposed network in an unsupervised manner. Experimental results on both simulated and real datasets demonstrate the effectiveness and efficiency of our method in comparison with the state-of-the-art water IOPs retrieving methods.


* **Input:** Hyperspectral Imagery(HSI) of ***optically deep water***
* **Output:** Inherent Optical Properties(IOPs) of the water, namely, ***absorption coefficient (a)*** , ***backward scattering coefficient (bb)***

<div align=center><img src="https://github.com/lijinchao98/IOPE_Net/blob/main/fig.jpg" width="600px" alt="IOPE-Net"></div>

The first part contains a IOPs estimation module, which is used for estimating absorption rate a(λ) and scattering rate bb(λ) from the input hyperspectral imagine concurrently. Consequently, the outputs of this part are the desired IOPs estimation results. The second part is designed for unsupervised training methodology. In this part, a spectrum reconstruction module has been established based on the classic bathymetric model, and we employ it to reconstruct sensor-observed spectrum with above IOPs estimation results. At the same time, the reconstruction errors evaluated subsequently are utilized to adjust the weight parameters of IOPs estimation module.

* **Rreference:** [2021, Qi, Hybrid Sequence Networks for Unsupervised Water Properties Estimation From Hyperspectral Imagery](https://doi.org/10.1109/JSTARS.2021.3068727). 
* The article proposes a hybrid sequence CNN+RNN, but in my implementation, CNN alone is sufficient and more mainly faster than adding RNN, so **I only use CNN** in my subsequent work.
* For the **Loss function**, here only use **MSE Loss** and **SA Loss**. **HMS Loss was not used**, yes, because I was too lazy to add it.
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


## IOPE-Net: 水体固有光学性质估计网络

从高光谱图像中产生对水的固有光学特性（IOPs）的准确估计在海洋勘探中起着重要作用。传统方法主要采用测深模型和数值优化算法来处理这个问题。然而，这些方法通常倾向于简化测深模型，缺乏描述参考光谱和估计光谱之间差异的能力，导致估计性能有限。为了得到一个更精确的结果，在这项工作中，我们提出了一个基于深度学习的新型网络来检索IOPs。该网络被命名为IOPs估计网络（IOPE-Net），它探索了一种混合序列结构来建立IOPs估计模块，以获取水体光谱的高维非线性特征来进行水体IOPs估计。此外，考虑到标记训练样本的不足，我们设计了一个频谱重建模块，结合经典的测深模型，以无监督的方式训练所提出的网络。在模拟和真实数据集上的实验结果证明了我们的方法与最先进的水体IOPs检索方法相比的有效性和效率。


* **输入:** 光学深水的高光谱图像
* **输出:** 水体固有光学性质，即，吸收系数a，后向散射系数bb

第一部分包含一个IOPs估计模块，用于从输入的高光谱图像中同时估计吸收率a（λ）和散射率bb（λ）。因此，这一部分的输出是所需的IOPs估计结果。第二部分是为无监督训练方法设计的。在这一部分中，我们建立了一个基于经典测深模型的光谱重建模块，并采用它来重建传感器观察到的光谱与上述IOPs估计结果。同时，随后评估的重建误差被用来调整IOPs估计模块的权重参数。

* **参考文献:** [2021, Qi, Hybrid Sequence Networks for Unsupervised Water Properties Estimation From Hyperspectral Imagery](https://doi.org/10.1109/JSTARS.2021.3068727). 
* 文章提出了CNN+RNN的混合序列，但在我的实现中，单单CNN就足够了，更主要的是比加入RNN更快，所以**我在后续工作中只使用CNN**。
* 对于**Loss函数，这里只使用**MSE Loss**和**SA Loss**。**没有使用**HMS损失，是的，因为我太懒了，没有添加它。
* 代码实现可能有一些不完善的地方，请随时提交问题。

## 环境需求
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## 数据格式
* 高光谱数据 (.hdr) 转换成 (.npy) 通过 'hdr_2_npy' 然后在输入网络前形状变为 (像元数, 波段数)。
* 所使用的数据都是反射率数据。

