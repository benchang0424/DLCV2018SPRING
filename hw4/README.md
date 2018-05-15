# Image Generation and Feature Disentanglement

## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Image Generation and Feature Disentanglement](#image-generation-and-feature-disentanglement)
	* [Table of Content](#table-of-content)
	* [Requirements](#requirements)
	* [Usage](#usage)
	* [Results](#results)
		

<!-- /code_chunk_output -->

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
    python3 plot.py --train_path <input directory> --out_path <output directory> \
                             --mode [vae, gan, acgan, <plot mode>]
    ```
    
    generates figures in output directory



## Results 