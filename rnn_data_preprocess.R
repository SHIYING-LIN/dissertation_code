
# CNN-RNN Data Pre-process
# Load useful packages
library(tensorflow)
library(keras)
library(dplyr)
library(purrr)
library(reticulate)
library(abind) # for merge arrays

# Load data
load('C:/Users/iandurbach/Desktop/dissertation/Data/MSc-2015-seal-data.RData')

# The number of frames within each target 
n_mat_per_target <- target.pix %>% map_depth(1, length) %>% unlist(use.names = T)

# Split sonar data by Seal/No Seal
Seals <- target.var[target.var$valid == "Seal",]
Seals$class <- 1
Targs <- target.var[target.var$valid != "Seal",]
Targs$class <- 0

nrow(Seals); nrow(Targs)

# Create pixel matrix and number of each target for Seals & Trags datasets
pixel_seals <- target.pix[Seals$id]
n_mat_seals <- n_mat_per_target[Seals$id]

pixel_targs <- target.pix[Targs$id]
n_mat_targs <- n_mat_per_target[Targs$id]


# Split into train/test/validation data
## Seal targets
set.seed(1377)
train_id_seals <- sample(1:nrow(Seals), round(0.8 * nrow(Seals)), replace = FALSE)
id <- setdiff(1:nrow(Seals), train_id_seals)
valid_id_seals <- sample(id, round(0.5 * length(id)), replace = FALSE)
test_id_seals <- setdiff(id, valid_id_seals)

print("-- Seal train/test/validation data ------")
cat(length(train_id_seals), "train samples\n")
cat(length(test_id_seals), "test samples\n")
cat(length(valid_id_seals), "validation sampels")
intersect(train_id_seals, valid_id_seals)

## Non-seal targets
set.seed(1377)
train_id_targs <- sample(1:nrow(Targs), round(0.8 * nrow(Targs)), replace = FALSE)
id <- setdiff(1:nrow(Targs), train_id_targs)
valid_id_targs <- sample(id, round(0.5 * length(id)), replace = FALSE)
test_id_targs <- setdiff(id, valid_id_targs)

print("------- No Seal train/test/validation data -------")
cat(length(train_id_targs), "train samples\n")
cat(length(test_id_targs), "test samples\n")
cat(length(valid_id_targs), "validation sampels")
intersect(train_id_targs, valid_id_targs)

# Add dataset labels of each target
#Seals$dataset <- c("train")
#Seals[test_id_seals,]$dataset <- c("test")
#Seals[valid_id_seals,]$dataset <- c("valid")




# Time step/maximum length of each sequence  
max_len <- 6

# ------------------ Seal Dataset -----------------------------
# Train data 
# The total length of training data
seq_len <- sum(n_mat_seals[train_id_seals]) - 
  length(n_mat_seals[train_id_seals]) * (max_len - 1)

# training data of the response 
y_train_seals <- array(rep(1, length = seq_len), dim = c(seq_len, 1))

# Remain the training targets' labels
y_train_seals_label <- data.frame(id = rep(names(n_mat_seals[train_id_seals]), 
                                  times = n_mat_seals[train_id_seals] - (max_len - 1))) %>%
  left_join(Seals, by = "id")
#nrow(y_train_seals_label)

# X
seq <- NULL

for (n in 1:length(train_id_seals)) {
  
  # length of i-th target
  for (i in train_id_seals[n]) {
    
    # sequence length of each target
    seq_per_len <- length(pixel_seals[[i]]) - (max_len - 1)
    # overall input sequence list 
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # initialize the overall input sequence and response sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                    dim = c(seq_per_len, max_len, 105, 105))
      
      # j-th row input length of i-th target
      for (j in 1:(length(pixel_seals[[i]]) - (max_len-1))) {
        
        # t-th time step of j-th row
        for (t in 1:max_len) {
          seq_[j,t,,] <- pixel_seals[[i]][[j+t-1]]
        }
      }
  }
  # save each element (sequence chunks of target) of input 
  # sequence in the pre-prepared list
  seq[[n]] <- seq_
}

#image(seq[[43]][1,2,,])

# Combine input sequences of seal targets (1-10, 11-20, 21-30, 31-40, 41-43)
cat(length(train_id_seals), "train samples (Seal)")
x_train_seals1 <- NULL; x_train_seals2 <- NULL; x_train_seals3 <- NULL
x_train_seals4 <- NULL; x_train_seals5 <- NULL; x_train_seals <- NULL

x_train_seals1 <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]], 
                  seq[[6]], seq[[7]], seq[[8]], seq[[9]], seq[[10]], along = 1)
x_train_seals2 <- abind(seq[[11]], seq[[12]],seq[[13]], seq[[14]], seq[[15]],
                  seq[[16]], seq[[17]], seq[[18]], seq[[19]], seq[[20]], along = 1)
x_train_seals3 <- abind(seq[[21]], seq[[22]], seq[[23]], seq[[24]], seq[[25]], 
                  seq[[26]], seq[[27]], seq[[28]], seq[[29]], seq[[30]], along = 1)
x_train_seals4 <- abind(seq[[31]], seq[[32]], seq[[33]], seq[[34]], seq[[35]],
                  seq[[36]], seq[[37]], seq[[38]], seq[[39]], seq[[40]], along = 1)
x_train_seals5 <- abind(seq[[41]], seq[[42]], seq[[43]], along = 1)

x_train_seals <- abind(x_train_seals1, x_train_seals2, x_train_seals3, 
                       x_train_seals4, x_train_seals5, along = 1)

# check dimension
cat("Dim_train_sequence (Seal):", dim(x_train_seals), "\n")
dim(x_train_seals)[1] == seq_len
#image(x_train_seals[1106,3,,])



## Test data -------------------------
# The total length of test data
seq_len <- sum(n_mat_seals[test_id_seals]) - 
  length(n_mat_seals[test_id_seals]) * (max_len - 1)

y_test_seals <- array(rep(1, length = seq_len), dim = c(seq_len, 1))

y_test_seals_label <- data.frame(id = rep(names(n_mat_seals[test_id_seals]), 
                                 times = n_mat_seals[test_id_seals] - (max_len - 1))) %>%
  left_join(Seals, by = "id")
nrow(y_test_seals_label)

# X
seq <- NULL

for (n in 1:length(test_id_seals)) {
  
  # prepare sequence list 
  for (i in test_id_seals[n]) {
    seq_per_len <- length(pixel_seals[[i]]) - (max_len - 1)
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # the overall input sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                  dim = c(seq_per_len, max_len, 105, 105)) 
    
    # j-th row of i-th target
    for (j in 1:(length(pixel_seals[[i]]) - (max_len-1))) {
      
      # t-th time step of j-th row
      for (t in 1:max_len) {
        seq_[j,t,,] <- pixel_seals[[i]][[j+t-1]]
      }
    }
  }
  # save each element (sequences of every target) of input sequence in the list
  seq[[n]] <- seq_
}

# Combine sequence of test targets
cat(length(test_id_seals), "test samples (Seal) \n")
x_test_seals <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]], along = 1)

# Check dimension
cat("Dim_test_sequence (Seal):", dim(x_test_seals), "\n")
dim(x_test_seals)[1] == seq_len


## Validation data ------------
# The total length of validation data
seq_len <- sum(n_mat_seals[valid_id_seals]) - 
  length(n_mat_seals[valid_id_seals]) * (max_len - 1)

y_valid_seals <- array(rep(1, length = seq_len), dim = c(seq_len, 1))

y_valid_seals_label <- data.frame(id = rep(names(n_mat_seals[valid_id_seals]), 
                                          times = n_mat_seals[valid_id_seals] - (max_len - 1))) %>%
  left_join(Seals, by = "id")
nrow(y_valid_seals_label)

# X
seq <- NULL

for (n in 1:length(valid_id_seals)) {
  
  # prepare sequence list 
  for (i in valid_id_seals[n]) {
    seq_per_len <- length(pixel_seals[[i]]) - (max_len - 1)
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # the overall input sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                  dim = c(seq_per_len, max_len, 105, 105)) 
    
    # the j-th row of i-th target
    for (j in 1:(length(pixel_seals[[i]]) - (max_len-1))) {
      
      # the t-th time step of j-th row
      for (t in 1:max_len) {
        seq_[j,t,,] <- pixel_seals[[i]][[j+t-1]]
      }
    }
  }
  # save each element (sequences of every target) of input sequence in the list
  seq[[n]] <- seq_
}

# Combine sequence of validation targets
cat(length(valid_id_seals), "validation samples (Seal) \n")
x_valid_seals <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]], seq[[6]], along = 1)

# Check dimension
cat("Dim_valid_sequence (Seal):", dim(x_valid_seals), "\n")
dim(x_valid_seals)[1] == seq_len

dim(x_valid_seals)[1] + dim(x_test_seals)[1] + dim(x_train_seals)[1] ==
  sum(n_mat_seals) - length(n_mat_seals) * (max_len - 1)




#------------------ Non-seal Targets Dataset -------------------------
## Train data 
seq_len <- sum(n_mat_targs[train_id_targs]) - 
  length(n_mat_targs[train_id_targs]) * (max_len - 1)

y_train_targs <- array(rep(0, length = seq_len), dim = c(seq_len, 1))

y_train_targs_label <- data.frame(id = rep(names(n_mat_targs[train_id_targs]), 
                                           times = n_mat_targs[train_id_targs] - (max_len - 1))) %>%
  left_join(Targs, by = "id")
nrow(y_train_targs_label)


# X
seq <- NULL
for (n in 1:length(train_id_targs)) {
  
  # length of i-th target
  for (i in train_id_targs[n]) {
    
    # sequence length of each target
    seq_per_len <- length(pixel_targs[[i]]) - (max_len - 1)
    # overall input sequence list 
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # initialize the overall input sequence and response sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                  dim = c(seq_per_len, max_len, 105, 105))
    
    # j-th row input length of i-th target
    for (j in 1:(length(pixel_targs[[i]]) - (max_len-1))) {
      
      # t-th time step of j-th row
      for (t in 1:max_len) {
        seq_[j,t,,] <- pixel_targs[[i]][[j+t-1]]
      }
    }
  }
  # save each element (sequence chunks of target) of input sequence in the list
  seq[[n]] <- seq_
}

# Combine input sequences of non-seal targets (1-10, 11-20, 21-30,
# 31-40, 41-40, 51-63)
cat(length(train_id_targs), "train sequences (No Seal)\n")
x_train_targs1 <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]], 
                        seq[[6]], seq[[7]], seq[[8]], seq[[9]], seq[[10]], along = 1)
x_train_targs2 <- abind(seq[[11]], seq[[12]],seq[[13]], seq[[14]], seq[[15]],
                        seq[[16]], seq[[17]], seq[[18]], seq[[19]], seq[[20]], along = 1)
x_train_targs3 <- abind(seq[[21]], seq[[22]], seq[[23]], seq[[24]], seq[[25]], 
                        seq[[26]], seq[[27]], seq[[28]], seq[[29]], seq[[30]], along = 1)
x_train_targs4 <- abind(seq[[31]], seq[[32]], seq[[33]], seq[[34]], seq[[35]],
                        seq[[36]], seq[[37]], seq[[38]], seq[[39]], seq[[40]], along = 1)
x_train_targs5 <- abind(seq[[41]], seq[[42]], seq[[43]], seq[[44]], seq[[45]],
                        seq[[46]], seq[[47]], seq[[48]], seq[[49]], seq[[50]], along = 1)
x_train_targs6 <- abind(seq[[51]], seq[[52]], seq[[53]], seq[[54]], seq[[55]],
                        seq[[56]], seq[[57]], seq[[58]], seq[[59]], seq[[60]], along = 1)
x_train_targs7 <- abind(seq[[61]], seq[[62]], seq[[63]], along = 1)

x_train_targs <- abind(x_train_targs1, x_train_targs2, x_train_targs3, 
                       x_train_targs4, x_train_targs5, x_train_targs6,
                       x_train_targs7, along = 1)

# check dimension
cat("Dim_train_sequence (No Seak):", dim(x_train_targs), "\n")
dim(x_train_targs)[1] == seq_len
#image(x_train_targs[1106,2,,])


## Test data -------------------------
seq_len <- sum(n_mat_targs[test_id_targs]) - 
  length(n_mat_targs[test_id_targs]) * (max_len - 1)

y_test_targs <- array(rep(0, length = seq_len), dim = c(seq_len, 1))
  
y_test_targs_label <- data.frame(id = rep(names(n_mat_targs[test_id_targs]), 
                                           times = n_mat_targs[test_id_targs] - (max_len - 1))) %>%
  left_join(Targs, by = "id")
nrow(y_test_targs_label)

# X
seq <- NULL

for (n in 1:length(test_id_targs)) {
  
  # prepare sequence list 
  for (i in test_id_targs[n]) {
    seq_per_len <- length(pixel_targs[[i]]) - (max_len - 1)
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # the overall input sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                  dim = c(seq_per_len, max_len, 105, 105)) 
    
    # j-th row of i-th target
    for (j in 1:(length(pixel_targs[[i]]) - (max_len-1))) {
      
      # t-th time step of j-th row
      for (t in 1:max_len) {
        seq_[j,t,,] <- pixel_targs[[i]][[j+t-1]]
      }
    }
  }
  # save each element (sequences of every target) of input sequence in the list
  seq[[n]] <- seq_
}

# Combine sequence of test targets
cat(length(test_id_targs), "test sequences (No Seal)\n")
x_test_targs <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]],
                      seq[[6]], seq[[7]], seq[[8]], along = 1)

# Check dimension
cat("Dim_test_sequence (No Seal):", dim(x_test_targs), "\n")
dim(x_test_targs)[1] == seq_len



# Validation data --------------------
seq_len <- sum(n_mat_targs[valid_id_targs]) - 
  length(n_mat_targs[valid_id_targs]) * (max_len - 1)

y_valid_targs <- array(rep(0, length = seq_len), dim = c(seq_len, 1))

y_valid_targs_label <- data.frame(id = rep(names(n_mat_targs[valid_id_targs]), 
                                          times = n_mat_targs[valid_id_targs] - (max_len - 1))) %>%
  left_join(Targs, by = "id")

# X
seq <- NULL

for (n in 1:length(valid_id_targs)) {
  
  # prepare sequence list 
  for (i in valid_id_targs[n]) {
    seq_per_len <- length(pixel_targs[[i]]) - (max_len - 1)
    seq[[n]] <- list(array(rep(0, seq_per_len * max_len * 105 * 105), 
                           dim = c(seq_per_len, max_len, 105, 105)))
    
    # the overall input sequence
    seq_ <- array(rep(0, seq_per_len  * max_len * 105 * 105),
                  dim = c(seq_per_len, max_len, 105, 105)) 
    
    # the j-th row of i-th target
    for (j in 1:(length(pixel_targs[[i]]) - (max_len-1))) {
      
      # the t-th time step of j-th row
      for (t in 1:max_len) {
        seq_[j,t,,] <- pixel_targs[[i]][[j+t-1]]
      }
    }
  }
  # save each element (sequences of every target) of input sequence in the list
  seq[[n]] <- seq_
}

# Combine sequence of validation targets
cat(length(valid_id_targs), "validation sequence (No Seal) \n")
x_valid_targs <- abind(seq[[1]], seq[[2]], seq[[3]], seq[[4]], seq[[5]],
                       seq[[6]], seq[[7]], seq[[8]], along = 1)

# Check dimension
cat("Dim_valid_sequence (No Seal):", dim(x_valid_targs), "\n")
dim(x_valid_targs)[1] == seq_len

dim(x_valid_targs)[1] + dim(x_test_targs)[1] + dim(x_train_targs)[1] ==
  sum(n_mat_targs) - length(n_mat_targs) * (max_len - 1)



# ------------------ True train/test/validation data ------------------------
# Release computer memory
remove(x_train_seals1, x_train_seals2, x_train_seals3, x_train_seals4,
       x_train_seals5, x_train_targs1, x_train_targs2, x_train_targs3,
       x_train_targs4, x_train_targs5, x_train_targs6, x_train_targs7)

# Combine sequences of all targets
x_train <- abind(x_train_seals, x_train_targs, along = 1)
x_test <- abind(x_test_seals, x_test_targs, along = 1)
x_valid <- abind(x_valid_seals, x_valid_targs, along = 1)

dim(x_train); dim(x_test); dim(x_valid)

# Normalization
x_train <- x_train / max(x_train)
x_test <- x_test / max(x_test)
x_valid <- x_valid / max(x_valid)

max(x_train); min(x_train)

# Add channel size
x_train <- array_reshape(x_train, dim = c(dim(x_train)[1], max_len, 105, 105, 1))
x_test <- array_reshape(x_test, dim = c(dim(x_test)[1], max_len, 105, 105, 1))
x_valid <- array_reshape(x_valid, dim = c(dim(x_valid)[1], max_len, 105, 105, 1))

remove(x_train_seals, x_train_targs, x_valid_seals, x_valid_targs, x_test_seals, x_test_targs)
remove(seq_, seq, target.pix.nopad)

## Y
y_train <- abind(y_train_seals, y_train_targs, along = 1)
y_test <- abind(y_test_seals, y_test_targs, along = 1)
y_valid <- abind(y_valid_seals, y_valid_targs, along = 1)

table(y_train); table(y_test); table(y_valid)

# Keep Y labels
train <- rbind(y_train_seals_label, y_train_targs_label)
valid <- rbind(y_valid_seals_label, y_valid_targs_label)
test <- rbind(y_test_seals_label, y_test_targs_label)

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 2)
y_test <- to_categorical(y_test, 2)
y_valid <- to_categorical(y_valid, 2)

cat("overall_input_sequence_shape:", dim(x_train) ,"\n")
cat("output_shape:", dim(y_train), "\n")

