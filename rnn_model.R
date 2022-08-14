# CNN-RNN model training & hyperparamter tuning 

# Model parameters
filters <- 32
learning_rate <- 0.001
batch_size <- 32
epochs <- 20

# Define model
history <- keras_model_sequential() %>% 
  # The pre-trained CNN
  time_distributed(layer_conv_2d(input_shape = c(max_len, 105, 105, 1),
                                 filters = filters, kernel_size = c(3,3), activation = "relu", padding = "same",
                                 kernel_initializer = 'glorot_uniform', 
                                 kernel_regularizer=regularizer_l2(0.001))) %>%
  time_distributed(layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = "relu",
                                 padding = "same", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizer_l2(0.001))) %>%
  time_distributed(layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = "relu", 
                                 padding = "same", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizer_l2(0.001))) %>%
  time_distributed(layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = "relu", 
                                 padding = "same", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizer_l2(0.001))) %>%
  # RNN with LSTM units  
  time_distributed(layer_flatten()) %>%
  layer_lstm(64, return_sequences = FALSE, dropout = 0.25) %>%
  layer_dense(2, activation = "softmax") 

# Compile model
history %>% compile(
  loss = "binary_crossentropy", 
  optimizer = optimizer_adam(learning_rate = learning_rate),
  metrics = c('accuracy')
)

# Model fitting
rnn <- history %>% fit(
  x_train,
  y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_valid, y_valid),
  # Control overfitting by early stopping
  callbacks = callback_early_stopping(
    monitor = "val_loss",
    patience = 5)
)


# Tune hyperparameters ------------------------------------------
# Load plot packages
library(ggplot2)
library(ggpubr)

# Set plot parameters
legend_theme <- theme(
  text = element_text(size = 20),
  legend.text = element_text(size = 20),
  legend.text.align = 0, 
  legend.key.width = unit(0.5,"inches"), 
  legend.title = element_text(size = 20),
  legend.position = "bottom"
)

axis_theme <- theme(axis.title = element_text(size = 20))

# The learning rate
p1 <- ggplot() +
  geom_line(aes(x = 1:length(rnn_lr_0.1$metrics$loss),
                y = rnn_lr_0.1$metrics$val_loss, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lr_0.01$metrics$loss),
                y = rnn_lr_0.01$metrics$val_loss, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn$metrics$loss),
                y = rnn$metrics$val_loss, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lr_0.0001$metrics$loss),
                y = rnn_lr_0.0001$metrics$val_loss, col = "4"), lwd = 1.2) +
  scale_color_discrete(name = "learning rate",
                       breaks = c("1", "2", "3", "4"),
                       labels = c("0.1", "0.01", "0.001", "0.0001")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  scale_y_continuous(limits = c(0.55,1.2)) +
  xlab("epoch") + ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme 

p2 <- ggplot() +
  geom_line(aes(x = 1:length(rnn_lr_0.1$metrics$loss),
                y = rnn_lr_0.1$metrics$val_accuracy, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lr_0.01$metrics$loss),
                y = rnn_lr_0.01$metrics$val_accuracy, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn$metrics$loss),
                y = rnn$metrics$val_accuracy, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lr_0.0001$metrics$loss),
                y = rnn_lr_0.0001$metrics$val_accuracy, col = "4"), lwd = 1.2) +
  scale_color_discrete(name = "learning rate",
                       breaks = c("1", "2", "3", "4"),
                       labels = c("0.1", "0.01", "0.001", "0.0001")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  scale_y_continuous(limits = c(0,0.80), breaks = seq(0,1,0.1)) +
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p1, p2, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")


# The maximum sequence length 
p3 <- ggplot() +
  geom_line(aes(x = 1:length(rnn_maxlen_2$metrics$loss),
                y = rnn_maxlen_2$metrics$val_loss, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_3$metrics$loss),
                y = rnn_maxlen_3$metrics$val_loss, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_4$metrics$loss),
                y = rnn_maxlen_4$metrics$val_loss, col = "4"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_5$metrics$loss),
                y = rnn_maxlen_5$metrics$val_loss, col = "5"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_6$metrics$loss),
                y = rnn_maxlen_6$metrics$val_loss, col = "6"), lwd = 1.2) +
  scale_color_discrete(name = "sequence length",
                       breaks = c("2", "3", "4", "5", "6"),
                       labels = c("2 timesteps", "3 timesteps", "4 timesteps", "5 timesteps", "6 timesteps")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  scale_y_continuous(limits = c(0.5,1.2), breaks = seq(0,1.5,0.1))+
  xlab("epoch") + ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme

p4 <- ggplot() +
  geom_line(aes(x = 1:length(rnn_maxlen_2$metrics$loss),
                y = rnn_maxlen_2$metrics$val_accuracy, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_3$metrics$loss),
                y = rnn_maxlen_3$metrics$val_accuracy, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_4$metrics$loss),
                y = rnn_maxlen_4$metrics$val_accuracy, col = "4"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_5$metrics$loss),
                y = rnn_maxlen_5$metrics$val_accuracy, col = "5"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_maxlen_6$metrics$loss),
                y = rnn_maxlen_6$metrics$val_accuracy, col = "6"), lwd = 1.2) +
  scale_color_discrete(name = "sequence length",
                       breaks = c("2", "3", "4", "5", "6"),
                       labels = c("2 timesteps", "3 timesteps", "4 timesteps", "5 timesteps", "6 timesteps")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  scale_y_continuous(limits = c(0.55,0.8), breaks = seq(0,1,0.05))+
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p3, p4, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")


# The number of the LSTM layers
p7 <- ggplot() +
  geom_line(aes(x = 1:length(rnn$metrics$loss),
                y = rnn$metrics$val_loss, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lstm_2$metrics$loss),
                y = rnn_lstm_2$metrics$val_loss, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lstm_3$metrics$loss),
                y = rnn_lstm_3$metrics$val_loss, col = "3"), lwd = 1.2) +
  scale_color_discrete(name = "the number of LSTM layers",
                       breaks = c("1", "2", "3"),
                       labels = c("1 layer", "2 layers", "3 layers")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  # scale_y_continuous(limits = c(0.6,1), breaks = seq(0,1,0.1))+
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

p8 <- ggplot() +
  geom_line(aes(x = 1:length(rnn$metrics$loss),
                y = rnn$metrics$val_accuracy, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lstm_2$metrics$loss),
                y = rnn_lstm_2$metrics$val_accuracy, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(rnn_lstm_3$metrics$loss),
                y = rnn_lstm_3$metrics$val_accuracy, col = "3"), lwd = 1.2) +
  scale_color_discrete(name = "the number of LSTM layers",
                       breaks = c("1", "2", "3"),
                       labels = c("1 layer", "2 layers", "3 layers")) +
  scale_x_continuous(limits = c(1,10), breaks = seq(0,10,2)) +
  scale_y_continuous(limits = c(0.5,0.8), breaks = seq(0,1,0.05))+
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p7, p8, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")

