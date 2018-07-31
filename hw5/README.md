# Action recognition

<!-- /code_chunk_output -->

## Task
  * Feature extraction from pre-trained CNN models
    * VGG-16
  * Trimmed action recognition
    * Training on RNN with sequences of CNN features and labels
  * Temporal action segmentation
    * Extend RNN model for sequence-to-sequence prediction

## Requirements
  * Python 3.6.4
  * numpy 1.14.2
  * scipy 1.0.1
  * Torch 0.4.0
  * torchvision 0.2.0
  * sklearn 0.19.1
  * skimage 0.14.0
  * skvideo 1.1.10

## Dataset
   * 29 Videos with frame size 240 * 320, 11 categories for label
   * For task 1 & 2, videos are trimmed into frames (3236 training samples / 517 validation samples)
   * For task 3, full video length is required (29 training samples / 5 validation samples)

## Usage

   * **Preprocessing**
    
      Refers to the [**preprocess**](https://github.com/benchang0424/DLCV2018SPRING/tree/master/hw5/preprocess) folder.
     
   * **Training**
     
   * **Testing**

     ```bash
      # Extract CNN-based feature and conduct prediction using average-pooled features
      bash hw5_p1.sh [directory of trimmed validation videos folder] [path of ground-truth csv file] [directory of output labels folder]

      # Extract CNN-based feature and conduct prediction through RNN
      bash hw5_p2.sh [directory of trimmed validation videos folder] [path of ground-truth csv file] [directory of output labels folder]

      # Whole video length action recognition
      bash hw5_p3.sh [directory of full-length validation videos folder] [directory of output labels folder]
     ```

## Results

   * **Accuracy**

      |         |CNN-based feautres           | RNN-based feautres  | Temporal action prediction
      | ------------- |:-------------:|:-----:|:-----:|
      | *Validation Accuracy.*    | 0.4952 | 0.5184 | 0.5885
  
   * **Visualization**
    
     * **CNN-based video features**
        <img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw5/output_imgs/CNN_tsne.jpg" width=700>
     
     * **RNN-based video features**
        <img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw5/output_imgs/RNN_tsne.jpg" width=700>
    
     * **Temporal action segmentation**
       * *OP01-R03-BaconAndEggs*

        <img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw5/output_imgs/OP01-R03-BaconAndEggs.jpg">

       * *OP06-R05-Cheeseburger*

        <img src="https://github.com/benchang0424/DLCV2018SPRING/blob/master/hw5/output_imgs/OP06-R05-Cheeseburger.jpg">
