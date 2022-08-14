
# CNN hyperparameters tuning
# Plot library
library(tidyverse)
library(ggplot2)
library(ggpubr)

# Set plot parameters
legend_theme <- theme(
  legend.text = element_text(colour = "black", size = 20),#????????????????????????
  legend.text.align = 0, #0???,1???,0.5??????, ?????????????????????????????????
  legend.key.width = unit(0.5,"inches"), #?????????????????????
  legend.title = element_text(colour = "black", size = 20),#??????????????????
  legend.position = "bottom"
)

axis_theme <- theme(
  axis.title = element_text(
  face = "plain", #??????("plain", "italic", "bold", "bold.italic")
  size = 20,
))


# The number of neurons in the convolution layers
cnn_conv_nodes_32 <- cnn

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
cnn_dense_nodes_64 <- cnn

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
cnn_dense_layer_3 <- cnn

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
cnn_rate_0.25 <- cnn

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



