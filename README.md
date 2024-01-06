# CIFAR-10 Image Classification Model

## Overview
This repository contains a TensorFlow-based image classification model designed to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset comprises 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Data
The model utilizes the CIFAR-10 dataset, which is directly loaded using TensorFlow's dataset library. The dataset is split into:
- `x_train, y_train`: Training data and labels.
- `x_test, y_test`: Testing data and labels.

## Model Architecture
The model is a convolutional neural network (CNN) with the following layers:
- Conv2D layers with ReLU activation for feature extraction.
- MaxPooling layers for downsampling.
- Flatten layer for converting 2D features to 1D.
- Dense layers with ReLU and sigmoid activation for classification.

## Preprocessing
The images are normalized by dividing pixel values by 255.0 to bring them into the range [0, 1].

## Model Training
The model is compiled with the following parameters:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metrics: Accuracy

It is trained for 20 epochs with a batch size of 32.

## Prediction
The model predicts the class of a given image. The image is preprocessed by resizing to 32x32 and normalizing. The predicted class is determined by finding the class index with the highest probability.

## Usage
To use the model:
1. Load the CIFAR-10 dataset using TensorFlow.
2. Normalize the image data.
3. Define and compile the CNN model.
4. Train the model with the training data.
5. Predict the class of new images using the trained model.

## Dependencies
- numpy
- tensorflow
- os (for file handling)

## Note
This code is specifically designed for the CIFAR-10 dataset and demonstrates basic image classification using CNNs. The model can be adapted or expanded for more complex image classification tasks.
