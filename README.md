<p align="center"><img src="assets/logo.png" width="480"\></p>

**This repository has gone stale as I unfortunately do not have the time to maintain it anymore. If you would like to continue the development of it as a collaborator send me an email at eriklindernoren@gmail.com.**

## PyTorch-GAN
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers. Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GANs to implement are very welcomed.

<b>See also:</b> [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#auxiliary-classifier-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [BEGAN](#began)
    + [BicycleGAN](#bicyclegan)
    + [Boundary-Seeking GAN](#boundary-seeking-gan)
    + [Cluster GAN](#cluster-gan)
    + [Conditional GAN](#conditional-gan)
    + [Context-Conditional GAN](#context-conditional-gan)
    + [Context Encoder](#context-encoder)
    + [Coupled GAN](#coupled-gan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#deep-convolutional-gan)
    + [DiscoGAN](#discogan)
    + [DRAGAN](#dragan)
    + [DualGAN](#dualgan)
    + [Energy-Based GAN](#energy-based-gan)
    + [Enhanced Super-Resolution GAN](#enhanced-super-resolution-gan)
    + [GAN](#gan)
    + [InfoGAN](#infogan)
    + [Least Squares GAN](#least-squares-gan)
    + [MUNIT](#munit)
    + [Pix2Pix](#pix2pix)
    + [PixelDA](#pixelda)
    + [Relativistic GAN](#relativistic-gan)
    + [Semi-Supervised GAN](#semi-supervised-gan)
    + [Softmax GAN](#softmax-gan)
    + [StarGAN](#stargan)
    + [Super-Resolution GAN](#super-resolution-gan)
    + [UNIT](#unit)
    + [Wasserstein GAN](#wasserstein-gan)
    + [Wasserstein GAN GP](#wasserstein-gan-gp)
    + [Wasserstein GAN DIV](#wasserstein-gan-div)

## Installation
    $ git clone https://github.com/cuteboyqq/GAN-Pytorch.git
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### Auxiliary Classifier GAN
_Auxiliary Classifier Generative Adversarial Network_

#### Authors
Augustus Odena, Christopher Olah, Jonathon Shlens

[[Paper]](https://arxiv.org/abs/1610.09585) [[Code]](implementations/acgan/acgan.py)

#### Run Example
```
$ cd implementations/acgan/
$ python3 acgan.py
```

<p align="center">
    <img src="assets/acgan.gif" width="360"\>
</p>

### Adversarial Autoencoder
_Adversarial Autoencoder_

#### Authors
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey

[[Paper]](https://arxiv.org/abs/1511.05644) [[Code]](implementations/aae/aae.py)

#### Run Example
```
$ cd implementations/aae/
$ python3 aae.py
```

### BEGAN
_BEGAN: Boundary Equilibrium Generative Adversarial Networks_

#### Authors
David Berthelot, Thomas Schumm, Luke Metz

[[Paper]](https://arxiv.org/abs/1703.10717) [[Code]](implementations/began/began.py)

#### Run Example
```
$ cd implementations/began/
$ python3 began.py
```

### BicycleGAN
_Toward Multimodal Image-to-Image Translation_

#### Authors
Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, Oliver Wang, Eli Shechtman


[[Paper]](https://arxiv.org/abs/1711.11586) [[Code]](implementations/bicyclegan/bicyclegan.py)

<p align="center">
    <img src="assets/bicyclegan_architecture.jpg" width="800"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../implementations/bicyclegan/
$ python3 bicyclegan.py
```

<p align="center">
    <img src="assets/bicyclegan.png" width="480"\>
</p>
<p align="center">
    Various style translations by varying the latent code.
</p>


### Boundary-Seeking GAN
_Boundary-Seeking Generative Adversarial Networks_

#### Authors
R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, Yoshua Bengio


[[Paper]](https://arxiv.org/abs/1702.08431) [[Code]](implementations/bgan/bgan.py)

#### Run Example
```
$ cd implementations/bgan/
$ python3 bgan.py
```

### Cluster GAN
_ClusterGAN: Latent Space Clustering in Generative Adversarial Networks_

#### Authors
Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, Sreeram Kannan


[[Paper]](https://arxiv.org/abs/1809.03627) [[Code]](implementations/cluster_gan/clustergan.py)

Code based on a full PyTorch [[implementation]](https://github.com/zhampel/clusterGAN).

#### Run Example
```
$ cd implementations/cluster_gan/
$ python3 clustergan.py
```

<p align="center">
    <img src="assets/cluster_gan.gif" width="360"\>
</p>


### Conditional GAN
_Conditional Generative Adversarial Nets_

#### Authors
Mehdi Mirza, Simon Osindero


[[Paper]](https://arxiv.org/abs/1411.1784) [[Code]](implementations/cgan/cgan.py)

#### Run Example
```
$ cd implementations/cgan/
$ python3 cgan.py
```

<p align="center">
    <img src="assets/cgan.gif" width="360"\>
</p>

### Context-Conditional GAN
_Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks_

#### Authors
Emily Denton, Sam Gross, Rob Fergus

[[Paper]](https://arxiv.org/abs/1611.06430) [[Code]](implementations/ccgan/ccgan.py)

#### Run Example
```
$ cd implementations/ccgan/
$ python3 ccgan.py
```

### Context Encoder
_Context Encoders: Feature Learning by Inpainting_

#### Authors
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros

[[Paper]](https://arxiv.org/abs/1604.07379) [[Code]](implementations/context_encoder/context_encoder.py)

#### Run Example
```
$ cd implementations/context_encoder/
<follow steps at the top of context_encoder.py>
$ python3 context_encoder.py
```

<p align="center">
    <img src="assets/context_encoder.png" width="640"\>
</p>
<p align="center">
    Rows: Masked | Inpainted | Original | Masked | Inpainted | Original
</p>

### Coupled GAN
_Coupled Generative Adversarial Networks_

#### Authors
Ming-Yu Liu, Oncel Tuzel


[[Paper]](https://arxiv.org/abs/1606.07536) [[Code]](implementations/cogan/cogan.py)

#### Run Example
```
$ cd implementations/cogan/
$ python3 cogan.py
```

<p align="center">
    <img src="assets/cogan.gif" width="360"\>
</p>
<p align="center">
    Generated MNIST and MNIST-M images
</p>

### CycleGAN
_Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_

#### Authors
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros


[[Paper]](https://arxiv.org/abs/1703.10593) [[Code]](implementations/cyclegan/cyclegan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_cyclegan_dataset.sh monet2photo
$ cd ../implementations/cyclegan/
$ python3 cyclegan.py --dataset_name monet2photo
```

<p align="center">
    <img src="assets/cyclegan.png" width="900"\>
</p>
<p align="center">
    Monet to photo translations.
</p>

### Deep Convolutional GAN
_Deep Convolutional Generative Adversarial Network_

#### Authors
Alec Radford, Luke Metz, Soumith Chintala


[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](implementations/dcgan/dcgan.py)

#### Run Example
```
$ cd implementations/dcgan/
$ python3 dcgan.py
```

<p align="center">
    <img src="assets/dcgan.gif" width="240"\>
</p>

### DiscoGAN
_Learning to Discover Cross-Domain Relations with Generative Adversarial Networks_

#### Authors
Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim


[[Paper]](https://arxiv.org/abs/1703.05192) [[Code]](implementations/discogan/discogan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan_architecture.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../implementations/discogan/
$ python3 discogan.py --dataset_name edges2shoes
```

<p align="center">
    <img src="assets/discogan.png" width="480"\>
</p>
<p align="center">
    Rows from top to bottom: (1) Real image from domain A (2) Translated image from <br>
    domain A (3) Reconstructed image from domain A (4) Real image from domain B (5) <br>
    Translated image from domain B (6) Reconstructed image from domain B
</p>

### DRAGAN
_On Convergence and Stability of GANs_

#### Authors
Naveen Kodali, Jacob Abernethy, James Hays, Zsolt Kira

[[Paper]](https://arxiv.org/abs/1705.07215) [[Code]](implementations/dragan/dragan.py)

#### Run Example
```
$ cd implementations/dragan/
$ python3 dragan.py
```

### DualGAN
_DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_

#### Authors
Zili Yi, Hao Zhang, Ping Tan, Minglun Gong


[[Paper]](https://arxiv.org/abs/1704.02510) [[Code]](implementations/dualgan/dualgan.py)


#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh facades
$ cd ../implementations/dualgan/
$ python3 dualgan.py --dataset_name facades
```

### Energy-Based GAN
_Energy-based Generative Adversarial Network_

#### Authors
Junbo Zhao, Michael Mathieu, Yann LeCun

[[Paper]](https://arxiv.org/abs/1609.03126) [[Code]](implementations/ebgan/ebgan.py)

#### Run Example
```
$ cd implementations/ebgan/
$ python3 ebgan.py
```

### Enhanced Super-Resolution GAN
_ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks_

#### Authors
Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang

#### Abstract
 The code is available at [this https URL](https://github.com/xinntao/ESRGAN).

[[Paper]](https://arxiv.org/abs/1809.00219) [[Code]](implementations/esrgan/esrgan.py)


#### Run Example
```
$ cd implementations/esrgan/
<follow steps at the top of esrgan.py>
$ python3 esrgan.py
```

<p align="center">
    <img src="assets/enhanced_superresgan.png" width="320"\>
</p>
<p align="center">
    Nearest Neighbor Upsampling | ESRGAN
</p>

### GAN
_Generative Adversarial Network_

#### Authors
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio


[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](implementations/gan/gan.py)

#### Run Example
```
$ cd implementations/gan/
$ python3 gan.py
```

<p align="center">
    <img src="assets/gan.gif" width="240"\>
</p>

### InfoGAN
_InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets_

#### Authors
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel



[[Paper]](https://arxiv.org/abs/1606.03657) [[Code]](implementations/infogan/infogan.py)

#### Run Example
```
$ cd implementations/infogan/
$ python3 infogan.py
```

<p align="center">
    <img src="assets/infogan.gif" width="360"\>
</p>
<p align="center">
    Result of varying categorical latent variable by column.
</p>

<p align="center">
    <img src="assets/infogan.png" width="360"\>
</p>
<p align="center">
    Result of varying continuous latent variable by row.
</p>

### Least Squares GAN
_Least Squares Generative Adversarial Networks_

#### Authors
Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley



[[Paper]](https://arxiv.org/abs/1611.04076) [[Code]](implementations/lsgan/lsgan.py)

#### Run Example
```
$ cd implementations/lsgan/
$ python3 lsgan.py
```


### MUNIT
_Multimodal Unsupervised Image-to-Image Translation_

#### Authors
Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz

 Code and pretrained models are available at [this https URL](https://github.com/nvlabs/MUNIT)

[[Paper]](https://arxiv.org/abs/1804.04732) [[Code]](implementations/munit/munit.py)

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../implementations/munit/
$ python3 munit.py --dataset_name edges2shoes
```

<p align="center">
    <img src="assets/munit.png" width="480"\>
</p>
<p align="center">
    Results by varying the style code.
</p>

### Pix2Pix
_Unpaired Image-to-Image Translation with Conditional Adversarial Networks_

#### Authors
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros



[[Paper]](https://arxiv.org/abs/1611.07004) [[Code]](implementations/pix2pix/pix2pix.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix_architecture.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh facades
$ cd ../implementations/pix2pix/
$ python3 pix2pix.py --dataset_name facades
```

<p align="center">
    <img src="assets/pix2pix.png" width="480"\>
</p>
<p align="center">
    Rows from top to bottom: (1) The condition for the generator (2) Generated image <br>
    based of condition (3) The true corresponding image to the condition
</p>

### PixelDA
_Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks_

#### Authors
Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, Dilip Krishnan



[[Paper]](https://arxiv.org/abs/1612.05424) [[Code]](implementations/pixelda/pixelda.py)

#### MNIST to MNIST-M Classification
Trains a classifier on images that have been translated from the source domain (MNIST) to the target domain (MNIST-M) using the annotations of the source domain images. The classification network is trained jointly with the generator network to optimize the generator for both providing a proper domain translation and also for preserving the semantics of the source domain image. The classification network trained on translated images is compared to the naive solution of training a classifier on MNIST and evaluating it on MNIST-M. The naive model manages a 55% classification accuracy on MNIST-M while the one trained during domain adaptation achieves a 95% classification accuracy.

```
$ cd implementations/pixelda/
$ python3 pixelda.py
```  
| Method       | Accuracy  |
| ------------ |:---------:|
| Naive        | 55%       |
| PixelDA      | 95%       |

<p align="center">
    <img src="assets/pixelda.png" width="480"\>
</p>
<p align="center">
    Rows from top to bottom: (1) Real images from MNIST (2) Translated images from <br>
    MNIST to MNIST-M (3) Examples of images from MNIST-M
</p>

### Relativistic GAN
_The relativistic discriminator: a key element missing from standard GAN_

#### Authors
Alexia Jolicoeur-Martineau




[[Paper]](https://arxiv.org/abs/1807.00734) [[Code]](implementations/relativistic_gan/relativistic_gan.py)

#### Run Example
```
$ cd implementations/relativistic_gan/
$ python3 relativistic_gan.py                 # Relativistic Standard GAN
$ python3 relativistic_gan.py --rel_avg_gan   # Relativistic Average GAN
```

### Semi-Supervised GAN
_Semi-Supervised Generative Adversarial Network_

#### Authors
Augustus Odena


[[Paper]](https://arxiv.org/abs/1606.01583) [[Code]](implementations/sgan/sgan.py)

#### Run Example
```
$ cd implementations/sgan/
$ python3 sgan.py
```

### Softmax GAN
_Softmax GAN_

#### Authors
Min Lin



[[Paper]](https://arxiv.org/abs/1704.06191) [[Code]](implementations/softmax_gan/softmax_gan.py)

#### Run Example
```
$ cd implementations/softmax_gan/
$ python3 softmax_gan.py
```

### StarGAN
_StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation_

#### Authors
Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo


[[Paper]](https://arxiv.org/abs/1711.09020) [[Code]](implementations/stargan/stargan.py)

#### Run Example
```
$ cd implementations/stargan/
<follow steps at the top of stargan.py>
$ python3 stargan.py
```

<p align="center">
    <img src="assets/stargan.png" width="640"\>
</p>
<p align="center">
    Original | Black Hair | Blonde Hair | Brown Hair | Gender Flip | Aged
</p>

### Super-Resolution GAN
_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_

#### Authors
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi



[[Paper]](https://arxiv.org/abs/1609.04802) [[Code]](implementations/srgan/srgan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/superresgan.png" width="640"\>
</p>

#### Run Example
```
$ cd implementations/srgan/
<follow steps at the top of srgan.py>
$ python3 srgan.py
```

<p align="center">
    <img src="assets/superresgan.png" width="320"\>
</p>
<p align="center">
    Nearest Neighbor Upsampling | SRGAN
</p>

### UNIT
_Unsupervised Image-to-Image Translation Networks_

#### Authors
Ming-Yu Liu, Thomas Breuel, Jan Kautz

#### Abstract
 Code and additional results are available in this [https URL](https://github.com/mingyuliutw/unit).

[[Paper]](https://arxiv.org/abs/1703.00848) [[Code]](implementations/unit/unit.py)

#### Run Example
```
$ cd data/
$ bash download_cyclegan_dataset.sh apple2orange
$ cd implementations/unit/
$ python3 unit.py --dataset_name apple2orange
```

### Wasserstein GAN
_Wasserstein GAN_

#### Authors
Martin Arjovsky, Soumith Chintala, Léon Bottou


[[Paper]](https://arxiv.org/abs/1701.07875) [[Code]](implementations/wgan/wgan.py)

#### Run Example
```
$ cd implementations/wgan/
$ python3 wgan.py
```

### Wasserstein GAN GP
_Improved Training of Wasserstein GANs_

#### Authors
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville



[[Paper]](https://arxiv.org/abs/1704.00028) [[Code]](implementations/wgan_gp/wgan_gp.py)

#### Run Example
```
$ cd implementations/wgan_gp/
$ python3 wgan_gp.py
```

<p align="center">
    <img src="assets/wgan_gp.gif" width="240"\>
</p>

### Wasserstein GAN DIV
_Wasserstein Divergence for GANs_

#### Authors
Jiqing Wu, Zhiwu Huang, Janine Thoma, Dinesh Acharya, Luc Van Gool



[[Paper]](https://arxiv.org/abs/1712.01026) [[Code]](implementations/wgan_div/wgan_div.py)

#### Run Example
```
$ cd implementations/wgan_div/
$ python3 wgan_div.py
```

<p align="center">
    <img src="assets/wgan_div.png" width="240"\>
</p>
