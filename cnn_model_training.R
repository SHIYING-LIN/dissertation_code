# Load useful packages
library(keras)
library(dplyr)
library(purrr)
library(reticulate)
library(abind) # for merge arrays

# Load data
load('C:/Users/iandurbach/Desktop/dissertation/Data/MSc-2015-seal-data.RData')

# ------------------------ Data preprocessing ------------------------------------ 
# Number of targets in total
length(target.pix)

# Have a look at the first frame of the first target
image(target.pix[[1]][[1]])

# The number of frames within each target 
n_mat_per_target <- target.pix %>% map_depth(1, length) %>% unlist(use.names = T)

# The total number of frames
n_mat <- sum(n_mat_per_target) 

# Dimension of each matrix/frame 
dim_mat <- dim(target.pix[[1]][[1]])

# Input parameters (width/height)
c(img_cols, img_rows) %<-% c(dim_mat[1], dim_mat[2])

# ---------------------------- For Seals ------------------------------------
print("---- For Seals ----")
Seals <- target.var[target.var$valid == "Seal",]
Seals$class <- ifelse(Seals$valid  == "Seal", 1, 0)

# Create the pixel matrix of all seals
pixel_seals <- target.pix[Seals$id]
n_mat_seals <- sum(n_mat_per_target[Seals$id])

x_mat_seals <- array(rep(0, n_mat_seals * dim_mat[1] * dim_mat[2]), 
                     dim = c(n_mat_seals, dim_mat[1], dim_mat[2]))
N <- 0
for(i in 1:length(pixel_seals)){
  for(j in 1:length(pixel_seals[[i]])){
    N <- N + 1
    x_mat_seals[N,,] <- pixel_seals[[i]][[j]]
  }
}

dim(x_mat_seals)

# X
# Train/test/validation data 
set.seed(1377)
train_id_seals <- sample(1:n_mat_seals, round(0.8 * n_mat_seals), replace = FALSE)
x_train_seals <- x_mat_seals[train_id_seals,,]

id <- setdiff(1:n_mat_seals, train_id_seals)
test_id_seals <- sample(id, round(0.5 * length(id)), replace = FALSE)
x_test_seals <- x_mat_seals[test_id_seals,,]

valid_id_seals <- setdiff(id, test_id_seals)
x_valid_seals <- x_mat_seals[valid_id_seals,,]

# Check the accuracy of data classification 
intersect(train_id_seals, test_id_seals)
intersect(test_id_seals, valid_id_seals)

cat(nrow(x_train_seals), "train samples",
    paste0("(", round(nrow(x_train_seals) / n_mat_seals * 100, 1), "%)"), "\n")
cat(nrow(x_valid_seals), "train samples",
    paste0("(", round(nrow(x_valid_seals) / n_mat_seals * 100, 1), "%)"), "\n")
cat(nrow(x_test_seals), "train samples",
    paste0("(", round(nrow(x_test_seals) / n_mat_seals * 100, 1), "%)"), "\n")

# Y
# Transform Y into the long format 
seals_class <- data.frame(id = rep(names(n_mat_per_target[Seals$id]), 
                                   times = n_mat_per_target[Seals$id])) %>%
  left_join(Seals, by = "id")

# Train/test/validation data
y_train_seals <- as.matrix(seals_class$class[train_id_seals], ncol = 1)
y_test_seals <- as.matrix(seals_class$class[test_id_seals], ncol = 1)
y_valid_seals <- as.matrix(seals_class$class[valid_id_seals], ncol = 1)


#------------------------  For Non-seal Targets   ----------------------------------------
print("---- For Non-seal Targets ----")
Targs <- target.var[target.var$valid != "Seal",]
Targs$class <- ifelse(Targs$valid  == "Seal", 1, 0)

# Create the pixel matrix of all non-seal targets
pixel_targs <- target.pix[Targs$id]
n_mat_targs <- sum(n_mat_per_target[Targs$id])

x_mat_targs <- array(rep(0, n_mat_targs * dim_mat[1] * dim_mat[2]), 
                     dim = c(n_mat_targs, dim_mat[1], dim_mat[2]))
N <- 0
for(i in 1:length(pixel_targs)){
  for(j in 1:length(pixel_targs[[i]])){
    N <- N + 1
    x_mat_targs[N,,] <- pixel_targs[[i]][[j]]
  }
}

dim(x_mat_targs)

# X
# Train/test/validation data 
set.seed(1377)
train_id_targs <- sample(1:n_mat_targs, round(0.8 * n_mat_targs), replace = FALSE)
x_train_targs <- x_mat_targs[train_id_targs,,]

id <- setdiff(1:n_mat_targs, train_id_targs)
test_id_targs <- sample(id, round(0.5 * length(id)), replace = FALSE)
x_test_targs <- x_mat_targs[test_id_targs,,]

valid_id_targs <- setdiff(id, test_id_targs)
x_valid_targs <- x_mat_targs[valid_id_targs,,]

# Check the accuracy of data classification 
intersect(train_id_targs, valid_id_targs)
cat(nrow(x_train_targs), "train samples",
    paste0("(", round(nrow(x_train_targs) / n_mat_targs * 100, 1), "%)"), "\n")
cat(nrow(x_valid_targs), "validation samples", 
    paste0("(", round(nrow(x_valid_targs) / n_mat_targs * 100, 1) ,"%)"), "\n")
cat(nrow(x_test_targs), "test samples", 
    paste0("(", round(nrow(x_test_targs) / n_mat_targs * 100, 1), "%)"), "\n")

# Y
# Transform Y into the long format 
targs_class <- data.frame(id = rep(names(n_mat_per_target[Targs$id]), 
                                   times = n_mat_per_target[Targs$id])) %>%
  left_join(Targs, by = "id")

# Train/test/validation data
y_train_targs <- as.matrix(targs_class$class[train_id_targs], ncol = 1)
y_test_targs <- as.matrix(targs_class$class[test_id_targs], ncol = 1)
y_valid_targs <- as.matrix(targs_class$class[valid_id_targs], ncol = 1)



# --------------------- True train/validation/test data ------------------------
# Merge X matrices from seal & non-seal targets
x_train <- abind(x_train_seals, x_train_targs, along = 1)
x_test <- abind(x_test_seals, x_test_targs, along = 1)
x_valid <- abind(x_valid_seals, x_valid_targs, along = 1)

# Reshape dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
x_valid <- array_reshape(x_valid, c(nrow(x_valid), img_rows, img_cols, 1))

# Transform RGB values into [0,1] range
max(x_train); min(x_train)
max.pix <- max(x_mat_seals, x_mat_targs)
x_train <- x_train / max.pix
x_test <- x_test / max.pix
x_valid <- x_valid / max.pix

cat("x_train_shape:", dim(x_train), "\n")
cat("x_test_shape:", dim(x_test), "\n")
cat("x_valid_shape:", dim (x_valid), "\n")

# Merge Y matrices from seal & non-seal targets
y_train <- rbind(y_train_seals, y_train_targs)
y_test <- rbind(y_test_seals, y_test_targs)
y_valid <- rbind(y_valid_seals, y_valid_targs)

table(y_train); table(y_test); table(y_valid)

# Keep the labels of each frame
y_train_labels <- y_train
y_test_labels <- y_test
y_valid_labels <- y_valid

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 2)
y_test <- to_categorical(y_test, 2)
y_valid <- to_categorical(y_valid, 2)

cat("y_train_shape:", dim(y_train), "\n")
cat("y_test_shape:", dim(y_test), "\n")
cat("y_valid_shape:", dim(y_valid), "\n")


#------------------- CNN ----------------------------------------
# hyperparameters 
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
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
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

plot(cnn)


# -------------------------------------------
# Evaluate model: This is frame-level accuracy in three data sets
# Train data
pred_train <- history %>% predict_classes(x_train)
train_table <- table(Predicted = pred_train, Actual = y_train_labels)
score_train <- history %>% evaluate(x_train, y_train)

# Validation data
pred_valid <- history %>% predict_classes(x_valid)
valid_table <- table(Predict = pred_valid, Actual = y_valid_labels)
score_valid <- history %>% evaluate(x_valid, y_valid)

# Model performance on test data  
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


# Plot of the model performance in three datasets
library(ggplot2)
library(tidyverse)

ggplot() +
  geom_bar(aes(cnn$metrics$))




# "target-level" accuracy (See cnn_prediction.R)



