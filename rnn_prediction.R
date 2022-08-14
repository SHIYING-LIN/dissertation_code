# CNN-RNN Prediction 

# ------------------------------- Target-level Accuracy ------------------------------- 
# Validation data 
# Merge the targets in the validation data with their actual labels
pred_class <- history %>% predict(x_valid) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
valid <- cbind(valid, pred_class)

# Tune thresholds
thre <- seq(0.05, 0.95, 0.05)
valid_table <- array(as.matrix(0, ncol = 2, nrow = 2), dim = c(length(thre), 2, 2))
valid_acc <- rep(0, length = length(thre))

N <- 0
for(i in thre) {
  N <- N + 1
  
  # Get the predicted class of each target 
  v <- valid %>% 
    group_by(id) %>% 
    summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
    # Predict each target's label by majority vote
    mutate(targ_class = ifelse(meanseal > i, 1, 0))
  
  # Create confusion matrix
  valid_table[N,,] <- table(Pred = v$targ_class, Actual = v$true_class)
  
  # Compute the target-level accuracy
  valid_acc[N] <- round((valid_table[N,1,1] + valid_table[N,2,2]) / sum(valid_table[N,,]) * 100, 2)
}

# Record validation accuracy with its corresponding threshold
v <- cbind(thre, valid_acc) %>% as.data.frame()

# The best threshold 
best_valid_thre <- thre[which.max(valid_acc)]

# Plot accuracy with the different thresholds
same_acc <- data.frame(thre = thre[valid_acc == max(valid_acc)], acc = valid_acc[valid_acc == max(valid_acc)]) # thresholds with the same accuracy
ggplot() +
  geom_line(aes(x = v$thre, y = v$valid_acc), lwd = 1.2, alpha = .8) +
  geom_line(aes(x = same_acc$thre, y = same_acc$acc), lwd = 1.5, col = "red") +
  geom_point(aes(x = same_acc$thre, y = same_acc$acc), size = 3, col = "red") +
  scale_x_continuous(limits = c(0.05,0.95), breaks = seq(0,1,0.05)) +
  scale_y_continuous(limits = c(45,86), breaks = seq(10,90,5)) +
  theme(legend.position = "none") +
  xlab("threshold") +
  ylab("accuracy (%)") +
  theme_bw() +
  theme(text = element_text(size = 25))

# The test result (accuracy & confusion matrix) using the best threshold
valid_result <- valid %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))
valid_table <- table(Pred = valid_result$targ_class, Actual = valid_result$true_class)
valid_acc <- round((valid_table[1,1] + valid_table[2,2]) / sum(valid_table) * 100, 2)


# Train data ------------------------------------------------------------
pred_class <- history %>% predict(x_train) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
train <- cbind(train, pred_class) 

# Get the predicted class by majority vote using the best threshold
train_result <- train %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))

# Create confusion matrix 
train_table <- table(Pred = train_result$targ_class, Actual = train_result$true_class)

# Calculate the target-level accuracy 
train_acc <- round((train_table[1,1] + train_table[2,2]) / sum(train_table) * 100, 2)


# Test data ------------------------------------------------------------
pred_class <-  history %>% predict(x_test) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
test <- cbind(test, pred_class)

# Get the predicted class by majority vote using the best threshold
test_result <- test %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0)) 

# Create confusion matrix 
test_table <- table(Pred = test_result$targ_class, Actual = test_result$true_class)

# Calculate the target-level accuracy 
test_acc <- round(((test_table[1,1] + test_table[2,2]) / sum(test_table)) * 100, 2)

# Output target-level accuracy in train & test & validation subsets
train_acc; valid_acc; test_acc

# Save the results
rnn_train_targ_acc <- train_acc
rnn_test_targ_acc <- test_acc
rnn_valid_targ_acc <- valid_acc


# ------------------------------ Frame-level accuracy ------------------------------
# Merge train & test & validation dataset
train$dataset <- c("train")
valid$dataset <- c("valid")
test$dataset <- c("test")
rnn_pred <- rbind(train, valid, test)

# Create the confusion matrices in three subsets at the frame level
test_table_frame <- table(Pred = test$class, Actual = test$pred_class)
valid_table_frame <- table(Pred = valid$class, Actual = valid$pred_class)
train_table_frame <- table(Pred = train$class, Actual = train$pred_class)

# Frame-level accuracy in the test set
round((test_table_frame[1,1] + test_table_frame[2,2]) / sum(test_table_frame) * 100, 2)

# false positive rate
round(test_table_frame[2,1] / sum(test_table_frame) * 100, 2)

# false negative rate
round(test_table_frame[1,2] / sum(test_table_frame) * 100, 2)

# Compare the accuracy at the target level and frame level
(rnn_train_targ_acc - rnn_train_frame_acc) / rnn_train_frame_acc * 100
(rnn_test_targ_acc - rnn_test_frame_acc) / rnn_test_frame_acc * 100
(rnn_valid_targ_acc - rnn_valid_frame_acc) / rnn_valid_frame_acc * 100


# ------------------------------ Frame-level accuracy per target -------------------------------------
rnn_pred <- rnn_pred[rnn_pred$dataset == "test",] # foucs on the model performance in the test set
correct_class_per_tagret <- NULL
rnn_target_acc_per_target <- NULL

for (id in c(unique(rnn_pred$id))) {
  correct_class_per_tagret[id] <- nrow(rnn_pred[rnn_pred$id == id & rnn_pred$class == rnn_pred$pred_class,])
  rnn_target_acc_per_target[id] <- round(correct_class_per_tagret[id] / nrow(rnn_pred[rnn_pred$id == id,]), 2)
}

# Combine targets with their ID
rnn_target_acc_per_target <- rnn_target_acc_per_target %>% unlist() %>% as.data.frame()
colnames(rnn_target_acc_per_target) <- c("rnn_target_acc")
rnn_target_acc_per_target <- cbind(id = unique(rnn_pred$id), rnn_target_acc_per_target)

# The misclassified target 
rnn_target_acc_per_target_mis <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc != 1,]

# Targets with 100% and 0% accuracy metrics
rnn_acc_100 <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc == 1,]
rnn_acc_0 <- nrow(rnn_target_acc_per_target[rnn_target_acc_per_target$target_acc == 0,])


# Plot the misclassification targets

# Change figure margins
par(mar = c(1,1,1,1))

# Check its actual label
first(rnn_pred$class[rnn_pred$id == "1809500048"]) 

# Look at the predicted labels
rnn_pred$pred_class[rnn_pred$id == "1809500048"]

length(target.pix[["1809500048"]])
par(mfrow = c(4,8))
for (i in 1:31) {image <- image(target.pix.nopad[["1809500048"]][[i]])}



