# CNN prediction

# --------------------- Target-level accuracy -------------------------------
# Validation data 
# Merge predictions of targets in the validation data with their actual label
valid <- rbind(seals_class[valid_id_seals,], targs_class[valid_id_targs,])
pred_valid_class <- history %>% predict_classes(x_valid) %>% as.data.frame()
colnames(pred_valid_class) <- c("pred_class")
valid <- cbind(valid, pred_valid_class)

# Tune threshold
thre <- seq(0.05, 0.95, 0.05)
valid_table <- array(as.matrix(0, ncol = 2, nrow = 2), dim = c(length(thre), 2, 2))
valid_acc <- rep(0, length = length(thre))

N <- 0
for(i in thre) {
  N <- N + 1
  
  # Get the predicted class of each target in the validation data
  v <- valid %>% 
    group_by(id) %>% 
    summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
    # Predict class of each target by majority vote
    mutate(targ_class = ifelse(meanseal > i, 1, 0))
  
  # Create confusion matrix
  valid_table[N,,] <- table(Pred = v$targ_class, Actual = v$true_class)
  
  # Calculate the target-level validation accuracy
  valid_acc[N] <- round((valid_table[N,1,1] + valid_table[N,2,2]) / sum(valid_table[N,,]) * 100, 2)
}

# Record validation accuracy with its corresponding threshold
v <- cbind(thre, valid_acc) %>% as.data.frame()

# The best threshold with the maximum accuracy
best_valid_thre <- thre[which.max(valid_acc)]

# Plot accuracy with different thresholds
same_acc <- data.frame(thre = thre[valid_acc == max(valid_acc)], acc = valid_acc[valid_acc == max(valid_acc)]) # thresholds with the same accuracy
ggplot() +
  geom_line(aes(x = v$thre, y = v$valid_acc), lwd = .8, alpha = .8) +
  geom_line(aes(x = same_acc$thre, y = same_acc$acc), lwd = 1.2, col = "red") +
  geom_point(aes(x = same_acc$thre, y = same_acc$acc), size = 2, col = "red") +
  scale_x_continuous(limits = c(0.05, 0.95), breaks = seq(0.05, 0.95, 0.05)) +
  theme(legend.position = "none") +
  xlab("threshold") +
  ylab("accuracy (%)") +
  theme_bw() +
  theme(axis.title = element_text(face = "plain", size = 12))
  
# The test result (accuracy & confusion matrix) using the best threshold
valid_result <- valid %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))

# The confusion matrix
valid_table <- table(Pred = valid_result$targ_class, Actual = valid_result$true_class)

# Target-level accuracy in the test set
valid_acc <- round((valid_table[1,1] + valid_table[2,2]) / sum(valid_table) * 100, 2)


# Train data ----------------------------------------
# Merge predictions of targets in the train data with their actual label
train <- rbind(seals_class[train_id_seals,], targs_class[train_id_targs,])
pred_train_class <- history %>% predict_classes(x_train) %>% as.data.frame()
colnames(pred_train_class) <- c("pred_class")
train <- cbind(train, pred_train_class)

# Get the predicted class of each target
train_result <- train %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))

# Confusion matrix
train_table<- table(Pred = train_result$targ_class, Actual = train_result$true_class)

# Target-level accuracy of the train data
train_acc <- round((train_table[1,1] + train_table[2,2]) / sum(train_table) * 100, 2)
cat("max_train_accuracy =", paste0(train_acc, "%"), "when threshold =", best_valid_thre)


# Test data ----------------------------------------
# Merge predictions of targets in the test data with their actual label
test <- rbind(seals_class[test_id_seals,], targs_class[test_id_targs,])
pred_test_class <- history %>% predict_classes(x_test) %>% as.data.frame()
colnames(pred_test_class) <- c("pred_class")
test <- cbind(test, pred_test_class)

# Get the predicted class of each target under best threshold
test_result <- test %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0)) 

# Confusion matrix
test_table <- table(Pred = test_result$targ_class, Actual = test_result$true_class)

# Target-level accuracy of the train data
test_acc <- round(test_table[1,1] + test_table[2,2] / sum(test_table) * 100, 2)

# Output the overall target-level accuracy of each set
cat("target-level accuracy in the training data:", train_acc, "\n")
cat("target-level accuracy in the test data:", test_acc, "\n")
cat("target-level accuracy in the validation data:", valid_acc, "\n")


# -------------------------- Frame-level accuracy ------------------------------
# Merge train & test & validation predictions
train$dataset <- c("train")
valid$dataset <- c("valid")
test$dataset <- c("test")
pred <- rbind(train, valid, test)

test_table_frame <- table(Pred = test$class, Actual = test$pred_class)
valid_table_frame <- table(Pred = valid$class, Actual = valid$pred_class)
train_table_frame <- table(Pred = train$class, Actual = train$pred_class)

# Frame-level accuracy in three subsets
round((test_table_frame[1,1] + test_table_frame[2,2]) / sum(test_table_frame) * 100, 2)
round((valid_table_frame[1,1] + valid_table_frame[2,2]) / sum(valid_table_frame) * 100, 2)
round((train_table_frame[1,1] + train_table_frame[2,2]) / sum(train_table_frame) * 100, 2)

# False positive rate
round(test_table_frame[2,1] / sum(test_table_frame) * 100, 2)
# False negative rate
round(test_table_frame[1,2] / sum(test_table_frame) * 100, 2)


# -------------------------- Frame-level accuracy per target --------------------------
pred <- pred[pred$dataset == "test"] # foucs on the test set
correct_class_per_tagret <- NULL
target_acc_per_target <- NULL

for (id in c(unique(pred$id))) {
  # The correct predicted class 
  correct_class_per_tagret[id] <- nrow(pred[pred$id == id & pred$class == pred$pred_class,])
  # The frame-level accuracy in each target
  target_acc_per_target[id] <- round(correct_class_per_tagret[id] / as.numeric(n_mat_per_target[id]), 2)
}

# Combine target with their ID
target_acc_per_target <- target_acc_per_target %>% unlist() %>% as.data.frame()
colnames(target_acc_per_target) <- c("target_acc")
target_acc_per_target <- cbind(id = unique(pred$id), target_acc_per_target)


# Check the misclassified target 
target_acc_per_target_mis <- target_acc_per_target[target_acc_per_target$target_acc != 1,]


# Let's have a look at the specific classification result by CNN!
# Targets with 100% accuracy
acc_100 <- target.var[target_acc_per_target$id[target_acc_per_target$target_acc == 1],]

# Check the target with lower accuracy (e.g. <0.5)
target_acc_per_target_mis <- target_acc_per_target[target_acc_per_target$target_acc < 0.5,]
nrow(target_acc_per_target_mis)

# Change figure margins
par("mar")
par(mar = c(1,1,1,1))

## 100% seal/seal 
length(target.pix[["2036190056"]])
par(mfrow = c(6,4))
for (i in 1:23) {
  image <- image(target.pix.nopad[["2036190056"]][[i]])
}

# 100% non-seal/non-seal
length(target.pix[["1735000011"]])
par(mfrow = c(4,4))
for (i in 1:13) {
  image <- image(target.pix.nopad[["1735000011"]][[i]])
}

# False negative rate example (reason 1)
first(pred$class[pred$id == "1809540102"]) # the actual class: seal
pred$pred_class[pred$id == "1809540102"] # the predicted class in each frame
target_acc_per_target_mis["1809540102", 2] # the frame-level accuracy

length(target.pix[["1809540102"]])
par(mfrow = c(4,4))
for (i in 1:16) {
  image <- image(target.pix.nopad[["1809540102"]][[i]])
}

# Another false negative rate example (reason 2)
target_acc_per_target_mis["1928490038", 2]
target_acc_per_target_mis["1928490038", 2] 

pred$pred_class[pred$id == "1928490038"] 
target.var$valid[target.var$id == "1928490038"]

# False positive rate exmaple 
length(target.pix[["1928490038"]])
par(mfrow = c(5,4))
for (i in 1:19) {
  image <- image(target.pix.nopad[["1928490038"]][[i]])
}

