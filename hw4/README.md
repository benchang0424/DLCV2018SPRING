# Image Generation and Feature Disentanglement

## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Image Generation and Feature Disentanglement](#image-generation-and-feature-disentanglement)
	* [Table of Content](#table-of-content)
  * [Task](#task)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [Results](#results)
		

<!-- /code_chunk_output -->
## Task
  * **Variational Autoencoder (VAE)**
  * **Generative Adversarial Network (GAN)**
  * **Auxiliary Classifier Generative Adversarial Network (ACGAN)**  

  For more details, please refers to the [PPT](https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw4/dlcv_hw4.pdf) provided by TAs.

## Requirements
  * Python 3.6
  * PyTorch 0.3.1
  * Keras 2.1.5
  * numpy
  * scikit-learn
  * skimage


## Usage

  * **Training VAE**

    ```
    python3 vae.py --train_path <path to ./hw4_data/>
    ```
  * **Training GAN**

    ```
    python3 gan.py --train_path <path to ./hw4_data/>
    ```
  * **Training ACGAN**

    ```
    python3 acgan.py --train_path <path to ./hw4_data/>
    ```
  * **Visualization of Inference Results**
    
    **Choose one argument in [...] list**
    ```
    python3 plot.py --train_path <input directory> --out_path <output directory> --mode [vae, gan, acgan]
    ```
    
    Will generate figures in output directory automatically.


## Results 
  * **Variational Autoencoder**
    * *Reconstruction* 

      <div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw4/figures/fig1_3.jpg">

    * *Random Generation*

      <div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw4/figures/fig1_4.jpg">

  * **Generative Adversial Network: *Random Generation***

    <div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw4/figures/fig2_3.jpg">

  * **Auxiliary Classifier GAN: *Smiling Generation [no smile/smile]***

    <div align=center><img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw4/figures/fig3_3.jpg">