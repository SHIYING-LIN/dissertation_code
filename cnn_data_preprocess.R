# CNN data preprocess

# Load useful packages
library(keras)
library(dplyr)
library(purrr)
library(reticulate)
library(abind) # for merge arrays

# Load data
load('C:/Users/iandurbach/Desktop/dissertation/Data/MSc-2015-seal-data.RData')

# Number of targets in total
length(target.pix)

# Have a look at sampled images
head(target.var)

## Images of non-seal & seal targets
par(mar = c(1,1,1,1))

length(target.pix[[1]])
par(mfrow = c(3,4))
for (i in 1:11) {image <- image(target.pix.nopad[[1]][[i]])}

length(target.pix[[2]])
par(mfrow = c(5,6))
for (i in 1:26) {image <- image(target.pix.nopad[[2]][[i]])}

# The number of frames within each target 
n_mat_per_target <- target.pix %>% map_depth(1, length) %>% unlist(use.names = T)

# The total number of frames
n_mat <- sum(n_mat_per_target) 

# Dimension of each matrix/frame 
dim_mat <- dim(target.pix[[1]][[1]])

# Input parameters (width/height)
c(img_cols, img_rows) %<-% c(dim_mat[1], dim_mat[2])

# ---------------------------- For Seals ------------------------------------
# Extract a subset with only seal targets
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

# X data
# Train/test/validation data 
set.seed(1377)
train_id_seals <- sample(1:n_mat_seals, round(0.8 * n_mat_seals), replace = FALSE)
x_train_seals <- x_mat_seals[train_id_seals,,]

id <- setdiff(1:n_mat_seals, train_id_seals)
test_id_seals <- sample(id, round(0.5 * length(id)), replace = FALSE)
x_test_seals <- x_mat_seals[test_id_seals,,]

valid_id_seals <- setdiff(id, test_id_seals)
x_valid_seals <- x_mat_seals[valid_id_seals,,]

# Check correctness of data splitting
intersect(train_id_seals, test_id_seals)
intersect(test_id_seals, valid_id_seals)
cat(nrow(x_train_seals), "train samples", "\n"))
cat(nrow(x_valid_seals), "train samples", "\n")
cat(nrow(x_test_seals), "train samples", "\n")

# Y data
# Transform Y into the long format 
seals_class <- data.frame(id = rep(names(n_mat_per_target[Seals$id]), 
                                   times = n_mat_per_target[Seals$id])) %>%
  left_join(Seals, by = "id")

# Train/test/validation data
y_train_seals <- as.matrix(seals_class$class[train_id_seals], ncol = 1)
y_test_seals <- as.matrix(seals_class$class[test_id_seals], ncol = 1)
y_valid_seals <- as.matrix(seals_class$class[valid_id_seals], ncol = 1)


#----------------------  For Non-seal Targets  --------------------------
# Extract a subset with only non-seal targets
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

# X data
# Train/test/validation data 
set.seed(1377)
train_id_targs <- sample(1:n_mat_targs, round(0.8 * n_mat_targs), replace = FALSE)
x_train_targs <- x_mat_targs[train_id_targs,,]

id <- setdiff(1:n_mat_targs, train_id_targs)
test_id_targs <- sample(id, round(0.5 * length(id)), replace = FALSE)
x_test_targs <- x_mat_targs[test_id_targs,,]

valid_id_targs <- setdiff(id, test_id_targs)
x_valid_targs <- x_mat_targs[valid_id_targs,,]

# Check correctness of data splitting
intersect(train_id_targs, valid_id_targs)
cat(nrow(x_train_targs), "train samples", "\n")
cat(nrow(x_valid_targs), "validation samples", "\n")
cat(nrow(x_test_targs), "test samples", "\n")

# Y data
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

# Display the dimension of X and Y data
cat("x_train_shape:", dim(x_train), "\n")
cat("x_test_shape:", dim(x_test), "\n")
cat("x_valid_shape:", dim (x_valid), "\n")

cat("y_train_shape:", dim(y_train), "\n")
cat("y_test_shape:", dim(y_test), "\n")
cat("y_valid_shape:", dim(y_valid), "\n")
