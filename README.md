# Deep learning for target classification in video images

This repository can be used to train deep learning for video image classification. All codes were implemented in R (4.2.0) using the TensorFlow library with Keras (version 2.9.0). We trained and tested all models on a virtual machine hosted on Microsoft Azure with 24 GB memory. The main packages covered dplyr (1.0.9), purrr (0.3.4), reticulate (1.25), ggplot (3.3.6), and abind (1.4-5).

Two main categories of deep learning methods, convolutional neural networks (CNNs) and recurrent neural networks (RNNs), were implemented for this target classification task. 


## Data preprocessing
A total of 574 min of sonar data (76GB) were collected, and 133 targets were manually detected for the classification classifier development. Before preprocessing data, the high-resolution images were transformed into pixel matrices, see sampled pixel images as follows.


The overall processes of data preprocessing for the CNN and CNN-RNN models were shown on the flow charts.




## Model training
 Details can be seen in `CNN_model_training` and `RNN_model_training` files.


## Hyperparameter tuning



## Predicting target's labels
