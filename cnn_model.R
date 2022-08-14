# CNN Model training & hyperparameter tuning

# Load library for plotting
library(tidyverse)
library(ggplot2)
library(ggpubr)

# Parameters -- set by users
input_shape <- c(105, 105, 1)
filters <- 32
dense_nodes <- 64
rate <- 0.25
batch_size <- 32
epochs <- 60

# Define model
history <- keras_model_sequential() %>%
  layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = rate) %>% 
  layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = rate) %>% 
  layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = rate) %>% 
  layer_conv_2d(filters = filters, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = rate) %>% 
  layer_flatten() %>% 
  layer_dense(units = dense_nodes, activation = 'relu') %>% 
  layer_dense(units = dense_nodes, activation = 'relu') %>% 
  layer_dense(units = dense_nodes, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 2, activation = 'softmax')
 
# Compile model
history %>% compile(
  loss =  "binary_crossentropy", 
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
  
# Model training
cnn <- history %>% fit(
  x_train,
  y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_valid, y_valid),
  # control over-fitting by early stopping
  callbacks = callback_early_stopping(
    monitor = "val_loss",
    patience = 5)
)

# Evaluate model
# Training data
pred_train <- history %>% predict_classes(x_train)
train_table <- table(Predicted = pred_train, Actual = y_train_labels)
score_train <- history %>% evaluate(x_train, y_train)

# Validation data
pred_valid <- history %>% predict_classes(x_valid)
valid_table <- table(Predict = pred_valid, Actual = y_valid_labels)
score_valid <- history %>% evaluate(x_valid, y_valid)

# Test data  
pred_test <- history %>% predict_classes(x_test)
test_table <- table(Predicted = pred_test, Actual = y_test_labels)
score_test <- history %>% evaluate(x_test, y_test)

# Output metrics
print("--------- 'frame-level' Accuracy ----------")
#cat('Train loss:', score_train[[1]], '\n')
cat('Train accuracy:', score_train[[2]], '\n')
#cat('Validation loss:', score_valid[[1]], '\n')
cat('Validation accuracy:', score_valid[[2]], '\n')
#cat('Test loss:', score_test[[1]], '\n')
cat('Test accuracy:', score_test[[2]], '\n')


# Set plot parameters
legend_theme <- theme(
  legend.text = element_text(colour = "black", size = 20),
  legend.text.align = 0, 
  legend.key.width = unit(0.5,"inches"),
  legend.title = element_text(colour = "black", size = 20),
  legend.position = "bottom"
)

axis_theme <- theme(
  axis.title = element_text(
    face = "plain", 
    size = 20
))


# The number of neurons in the convolution layers
p1 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_conv_nodes_32$metrics$loss), 
                y = cnn_conv_nodes_32$metrics$val_loss, col = "32"), lwd =1.2) +
  geom_line(aes(x = 1:length(cnn_conv_nodes_64$metrics$loss),
                y = cnn_conv_nodes_64$metrics$val_loss, col = "64"), lwd =1.2) +
  geom_line(aes(x = 1:length(cnn_conv_nodes_128$metrics$loss),
                y = cnn_conv_nodes_128$metrics$val_loss, col = "128"), lwd = 1.2) +
  scale_color_discrete(name = "the numebr of neurons in the convolution layers",
                       breaks = c("32", "64", "128"),
                       labels = c("N = 32", "N = 64", "N = 128")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.4, 0.7)) +
  xlab("epoch") + ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme

p2 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_conv_nodes_32$metrics$loss), 
                y = cnn_conv_nodes_32$metrics$val_accuracy, col = "32"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_conv_nodes_64$metrics$loss),
                y = cnn_conv_nodes_64$metrics$val_accuracy, col = "64"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_conv_nodes_128$metrics$loss), 
                y = cnn_conv_nodes_128$metrics$val_accuracy, col = "128"), lwd = 1.2) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.55, 0.81)) +
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p1, p2, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")


# The number of neurons in the dense layers
p3 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_dense_nodes_32$metrics$loss),
                y = cnn_dense_nodes_32$metrics$val_loss, col = "32"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_nodes_64$metrics$loss), 
                y = cnn_dense_nodes_64$metrics$val_loss, col = "64"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_nodes_128$metrics$loss),
                y = cnn_dense_nodes_128$metrics$val_loss, col = "128"), lwd = 1.2) +
  scale_color_discrete(name = "the numebr of neurons in the dense layers",
                       breaks = c("32", "64", "128"),
                       labels = c("N = 32", "N = 64", "N = 128")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.40, 0.70)) +
  xlab("epoch") + 
  ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme

p4 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_dense_nodes_32$metrics$loss),
                y = cnn_dense_nodes_32$metrics$val_accuracy, col = "32"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_nodes_64$metrics$loss), 
                y = cnn_dense_nodes_64$metrics$val_accuracy, col = "64"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_nodes_128$metrics$loss),
                y = cnn_dense_nodes_128$metrics$val_accuracy, col = "128"), lwd = 1.2) +
  scale_color_discrete(name = "the numebr of neurons in the dense layers",
                       breaks = c("32", "64", "128"),
                       labels = c("N = 32", "N = 64", "N = 128")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.55, 0.81)) +
  xlab("epoch") + 
  ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p3, p4, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")

# The number of dense layers
p5 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_dense_layer_1$metrics$loss),
                y = cnn_dense_layer_1$metrics$val_loss, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_2$metrics$loss), 
                y = cnn_dense_layer_2$metrics$val_loss, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_3$metrics$loss), 
                y = cnn_dense_layer_3$metrics$val_loss, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_4$metrics$loss), 
                y = cnn_dense_layer_4$metrics$val_loss, col = "4"), lwd = 1.2) +
  scale_color_discrete(name = "the numebr of dense layers",
                       breaks = c("1", "2", "3", "4"),
                       labels = c("1 layer", "2 layers", "3 layers", "4 layers")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.40, 0.70)) + 
  xlab("epoch") + 
  ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme

p6 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_dense_layer_1$metrics$loss),
                y = cnn_dense_layer_1$metrics$val_accuracy, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_2$metrics$loss),
                y = cnn_dense_layer_2$metrics$val_accuracy, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_3$metrics$loss),
                y = cnn_dense_layer_3$metrics$val_accuracy, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_dense_layer_4$metrics$loss), 
                y = cnn_dense_layer_4$metrics$val_accuracy, col = "4"), lwd = 1.2) +
  scale_color_discrete(name = "the numebr of dense layers",
                       breaks = c("1", "2", "3", "4"),
                       labels = c("1 layer", "2 layers", "3 layers", "4 layers")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.55, 0.82)) + 
  xlab("epoch") +
  ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p5, p6, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")

# Dropout rate
p7 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_rate_0.1$metrics$loss),
                y = cnn_rate_0.1$metrics$val_loss, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.15$metrics$loss),
                y = cnn_rate_0.15$metrics$val_loss, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.2$metrics$loss), 
                y = cnn_rate_0.2$metrics$val_loss, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.25$metrics$loss), 
                y = cnn_rate_0.25$metrics$val_loss, col = "4"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.3$metrics$loss), 
                y = cnn_rate_0.3$metrics$val_loss, col = "5"), lwd = 1.2) +
  scale_color_discrete(name = "dropout rate",
                       breaks = c("1", "2", "3", "4", "5"),
                       labels = c("rate = 0.1", "rate = 0.15", "rate = 0.2", "rate = 0.25", "rate = 0.3")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.40, 0.70)) +
  xlab("epoch") + ylab("validation loss") +
  theme_bw() + axis_theme + legend_theme

p8 <- ggplot() +
  geom_line(aes(x = 1:length(cnn_rate_0.1$metrics$loss),
                y = cnn_rate_0.1$metrics$val_accuracy, col = "1"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.15$metrics$loss), 
                y = cnn_rate_0.15$metrics$val_accuracy, col = "2"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.2$metrics$loss),
                y = cnn_rate_0.2$metrics$val_accuracy, col = "3"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.25$metrics$loss),
                y = cnn_rate_0.25$metrics$val_accuracy, col = "4"), lwd = 1.2) +
  geom_line(aes(x = 1:length(cnn_rate_0.3$metrics$loss), 
                y = cnn_rate_0.3$metrics$val_accuracy, col = "5"), lwd = 1.2) +
  scale_color_discrete(name = "dropout rate",
                       breaks = c("1", "2", "3", "4", "5"),
                       labels = c("rate = 0.1", "rate = 0.15", "rate = 0.2", "rate = 0.25", "rate = 0.3")) +
  scale_x_continuous(limits = c(0, 60)) +
  scale_y_continuous(limits = c(0.55, 0.81)) +
  xlab("epoch") + ylab("validation accuracy") +
  theme_bw() + axis_theme + legend_theme

ggarrange(p7, p8, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
