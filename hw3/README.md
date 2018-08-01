# Semantic Segmentation

## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Semantic Segmentation](#semantic-segmentation)
	* [Table of Content](#table-of-content)
	* [Requirements](#requirements)
	* [Usage](#usage)
	* [Results](#results)
		

<!-- /code_chunk_output -->

## Requirements
  * Python 3.6
  * Tensorflow 1.6
  * Keras 2.1.5
  * numpy
  * scipy

## Usage

  * **Training**

    ```
    python3 train.py --mode [fcn32s, fcn16s, fcn82] --train [path to ./hw3-train-validation/train/] --val [path to ./hw3-train-validation/validation/] 
    ```

  * **Testing**

    ```
    python3 test.py --model [model path] --val [input directory] --pred [output directory] 
    ```

  * **Calculate Mean IOU**

    ```
    python3 mean_iou_evaluate.py -g [ground truth directory] -p [predict directory]
    ```

## Results

  *	**Baseline Preformance**
    
    class     | FCN32s  | FCN16s  | FCN8s
    :--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
    class 0   | 0.71774 | 0.74861 | 0.75908
    class 1   | 0.87164 | 0.86890 | 0.87564
    class 2   | 0.25933 | 0.29304 | 0.34733
    class 3   | 0.74692 | 0.76598 | 0.78900
    class 4   | 0.70425 | 0.74807 | 0.73500
    class 5   | 0.65134 | 0.64684 | 0.64123
    mean_IoU  | **0.658988** | **0.678574** | **0.691213**
