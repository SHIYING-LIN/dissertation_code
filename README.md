# Deep learning for target classification in video images

This repository is designed to train deep learning classifiers for automatic target classification. All codes are implemented in R (4.2.0) using the TensorFlow library with Keras (version 2.9.0). We trained and tested models on a virtual machine hosted on Microsoft Azure with 24 GB memory. The main packages cover dplyr (1.0.9), purrr (0.3.4), reticulate (1.25), ggplot (3.3.6), and abind (1.4-5).

Two main categories of deep learning methods, convolutional neural networks (CNNs) and recurrent neural networks (RNNs), are implemented for the target classification task. 

- `Frame-only CNN`: model sees each frame as an individual input. It ignores temporal information in the video and consists of stacked convolutional layers, max-pooling layers, and dropout layers.

- `CNN-RNN`: model regards a clip of frames as a single input, which is produced by the pre-trained CNN and followed by the RNN with the LSTM layer.


## Data preprocessing
A total of 574 min of sonar data (76GB) were collected, and 133 targets were manually detected for the classification classifier development. Before preprocessing data, the high-resolution video data were converting into grayscale images frame-by-frame (also called "frames"). The sampled pixel images are as follows: the upper panel shows all frames from a non-seal target, and the lower panel shows all frames from a seal target.

![image](https://user-images.githubusercontent.com/91959615/184532255-84585fdc-c1d2-4e27-9c0c-6dfea8d22c6a.png)

![image](https://user-images.githubusercontent.com/91959615/184532243-04048f5c-faaf-490d-b28d-2912d58a03a7.png)


The following flowcharts illustrate the entire process of the two models predicting each target's label. The general process of the frame-only CNN model is shown in the upper panel and the process of CNN-RNN is shown in the lower panel.

![sonar data](https://user-images.githubusercontent.com/91959615/184537713-508fe410-799c-4bdf-8a8b-0a5dab4e5ac2.jpg)

![sonar data_rnn](https://user-images.githubusercontent.com/91959615/184537725-e5e5c885-93f0-473b-bb95-31444e146abd.jpg)


## Model training & Hyperparameter tuning
The frame-only CNN and CNN-RNN models can be trained and fine-tuned using the `CNN_model_training` and `RNN_model_training`, respectively. As for hyperparameter tuning, the grid search method lets users specify experiment parameter ranges. However, we do not include the code for grid search because we manually changed the values of the hyperparameters one by one. The loop can be used to tackle this low-efficient approach.

Here is an explanation of the parameters that can be used for an experiment:

- `filters`: the number of neurons in the convolutional layers (same applied throughout the different convolutional layers, good default is 32)
- `dense_nodes`: the number of neurons in the dense layers (same applied throughout the different dense layers, good default is 64)
- `rate`: the dropout rate of the dropout layers (same applied throughout the different dropout layers, good default is 0.25)
- `batch_size`:  the number of units manufactured in a production run (default to 32)
- `epochs`: the number of passes of the entire training dataset the algorithm has completed (default to 60 for CNN, 20 for CNN-RNN)
- `sequence_length` number of frames in one input sequence for CNN-RNN (good default is 6)


## Model prediction
The `cnn_prediction` and `rnn_prediction` file can be used to load a trained model and use it to output predicted labels for each frame in the dataset.




