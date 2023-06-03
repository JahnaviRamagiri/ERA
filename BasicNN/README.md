# Assignment 5 - README

## Overview
This assignment focuses on organizing code for a basic Convolutional Neural Network (CNN) trained on the MNIST dataset. The code has been divided into three files: `model.py`, `utils.py`, and `S5.ipynb`. This repository aims to provide a clear and structured implementation of the CNN model, along with utility functions and a main script for training and evaluation.

## Code Files

### model.py
This file contains the implementation of the CNN model architecture. It includes the definition of the neural network, consisting of convolutional layers, pooling layers, and fully connected layers. The file also contains functions for model initialization and model summary.

### utils.py
The `utils.py` file houses various utility functions utilized in the CNN implementation. It includes functions for data preprocessing, data loading, evaluation metrics, and other helper functions required for training and testing the model.

### S5.ipynb
The `S5.ipynb` notebook is the main script that brings together the code from `model.py` and `utils.py`. This notebook provides an end-to-end pipeline for training and evaluating the CNN model on the MNIST dataset. It consists of code cells with explanations and comments to guide you through the different steps of the process.


## Usage

To successfully run the code and reproduce the results:

1. Clone the repository to your local machine.
```
git clone https://github.com/your-username/assignment-5.git
```
3. Ensure that you have the necessary dependencies installed (e.g., Python, TensorFlow, etc.).
4. Open the `S5.ipynb` notebook using Jupyter Notebook or any compatible Python IDE.
5. Execute the code cells in sequential order to train and evaluate the CNN model.
6. Feel free to modify hyperparameters or experiment with different configurations as desired.
7. Explore the code in `model.py` and `utils.py` to understand the detailed implementations and utility functions.

**NOTE**: Change the import paths for model.py and utils.py according to their location on your local device.

```python
import utils
import model
```

# Code
## MNIST Dataset
The MNIST dataset is a collection of handwritten digits widely used for training and evaluating machine learning models. It consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

![MNIST DATASET](https://github.com/JahnaviRamagiri/ERA/assets/61361874/1eb98597-d620-4558-97a5-907c6a1256aa)


## Train Transforms and Train Loader
Before training the CNN model, the training data undergoes certain transformations to enhance the learning process. Transforms applied include:
- Random CenterCrop
- Random Rotation
- Normalization (mean subtraction and division by standard deviation)

The transformed training data is loaded into batches using a train loader with a specified batch size. The batch size determines the number of images processed in each iteration during training.

## Model Architecture
The model used for this assignment has the following layers and parameters:

![image](https://github.com/JahnaviRamagiri/ERA/assets/61361874/1347d272-da22-4653-9ab8-0cf2d9720128)


## Model Training
The training process involves iterating over the training data in batches and updating the model. Our model is trained on 20 epochs with a batch size of 32.

![image](https://github.com/JahnaviRamagiri/ERA/assets/61361874/8fbaa4bb-7fa9-40b6-8543-d35d8dd1bd11)

## Model Evaluation
The train and test losses and accuracies are stored to evaluate the model performance.

![image](https://github.com/JahnaviRamagiri/ERA/assets/61361874/da9c0f5e-0003-4262-86bc-c00bc9080b49)





