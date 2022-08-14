
# Target-level Accuracy 

library(ggplot2)

# Overall target-level accuracy ------------------------------
# Validation data 
# Prepare validation data for target accuracy
valid <- rbind(seals_class[valid_id_seals,], targs_class[valid_id_targs,])
pred_valid_class <- history %>% predict_classes(x_valid) %>% as.data.frame()
colnames(pred_valid_class) <- c("pred_class")
valid <- cbind(valid, pred_valid_class)

# Set parameters for calculating accuracy under different thresholds
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
    mutate(targ_class = ifelse(meanseal > i, 1, 0))
  
  # Confusion matrix
  valid_table[N,,] <- table(Pred = v$targ_class, Actual = v$true_class)
  
  # Calculate the validation accuracy
  valid_acc[N] <- round((valid_table[N,1,1] + valid_table[N,2,2]) / sum(valid_table[N,,]) * 100, 2)
}

# Record validation accuracy with its corresponding threshold
v <- cbind(thre, valid_acc) %>% as.data.frame()
best_valid_thre <- thre[which.max(valid_acc)]

# Plot accuracy with different thresholds
same_acc <- data.frame(thre = thre[valid_acc == max(valid_acc)], acc = valid_acc[valid_acc == max(valid_acc)])
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
  

# The test result (accuracy & confusion matrix) under the best threshold
valid_result <- valid %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))
valid_table <- table(Pred = valid_result$targ_class, Actual = valid_result$true_class)
valid_acc <- round((valid_table[1,1] + valid_table[2,2]) / sum(valid_table) * 100, 2)



# Train data ------------------------------------------------
# Prepare train data for prediction 
train <- rbind(seals_class[train_id_seals,], targs_class[train_id_targs,])
pred_train_class <- history %>% predict_classes(x_train) %>% as.data.frame()
colnames(pred_train_class) <- c("pred_class")
train <- cbind(train, pred_train_class)

# get the predicted class of each target
train_result <- train %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))

# confusion matrix
train_table<- table(Pred = train_result$targ_class, Actual = train_result$true_class)

# target-level accuracy under each threshold
train_acc <- round((train_table[1,1] + train_table[2,2]) / sum(train_table) * 100, 2)
cat("max_train_accuracy =", paste0(train_acc, "%"), "when threshold =", best_valid_thre)



# Test data ------------------------------------------------
# Prepare test data for prediction 
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

# Target-level accuracy 
test_acc <- round(test_table[1,1] + test_table[2,2] / sum(test_table) * 100, 2)
#cat("max_test_accuracy =", paste0(test_acc, "%"), "when threshold =", best_valid_thre)

train_acc
test_acc
valid_acc
(train_acc - cnn_score_train[[2]]*100)/cnn_score_train[[2]]*100
(valid_acc - cnn_score_valid[[2]]*100)/cnn_score_valid[[2]]*100

# Target-level accuracy per target ------------------------------
# Merge train & test & validation dataset
train$dataset <- c("train")
valid$dataset <- c("valid")
test$dataset <- c("test")
pred <- rbind(train, valid, test)
length(unique(pred$id)) == 133 # the unique label of all data should be 133


# Total frame-level accuracy: nrow(predicted class == target_class$valid) / 3478
frame_acc <- nrow(pred[pred$class == pred$pred_class,]) / n_mat
cat('Total frame-level accuracy:', paste0(round(frame_acc * 100, 2), "%"), '\n')


# Target-level accuracy per target
correct_class_per_tagret <- NULL
target_acc_per_target <- NULL

for (id in c(unique(pred$id))) {
  # the correct predicted class 
  correct_class_per_tagret[id] <- nrow(pred[pred$id == id & pred$class == pred$pred_class,])
  # the target-level accuracy in each target
  target_acc_per_target[id] <- round(correct_class_per_tagret[id] / as.numeric(n_mat_per_target[id]), 2)
}

# Keep the accuracy and add id
target_acc_per_target <- target_acc_per_target %>% unlist() %>% as.data.frame()
colnames(target_acc_per_target) <- c("target_acc")
target_acc_per_target <- cbind(id = unique(pred$id), target_acc_per_target)

# Save the frame-level accuracy per target in csv file
#write.csv(target_acc_per_target, file = "C:/Users/iandurbach/Desktop/per_target_accuracy.csv")



# The distribution of frame accuracy 
ggplot(target_acc_per_target, aes(target_acc * 100)) +
  geom_freqpoly(bins = 40, fill = "steelblue") +
 # geom_histogram(bins = 25) +
  theme_bw() +
  xlab("target-level accuracy (%)") 
#  scale_y_continuous(limits = c(0,60))


# Check the misclassified target 
target_acc_per_target_mis <- target_acc_per_target[target_acc_per_target$target_acc != 1,]

# The number of type two error (nonseal-seal)
nrow(target_acc_per_target_mis[target.var$valid[target.var$id == target_acc_per_target_mis$id] == "Seal",])


# -------------------------------------------------
# Let's have a look at which targets are prefect classified by CNN!

# Targets with 100% accuracy
acc_100 <- target.var[target_acc_per_target$id[target_acc_per_target$target_acc == 1],]
# the proportion of seals accounts for 100% accuracy
acc_100_seal <- nrow(acc_100[acc_100$valid == "Seal",]) / nrow(acc_100)
acc_100_seal

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

## 100% non-seal/non-seal
length(target.pix[["1735000011"]])
par(mfrow = c(4,4))
for (i in 1:13) {
  image <- image(target.pix.nopad[["1735000011"]][[i]])
}


## Actual: seal/ Predict: non-seal
first(pred$class[pred$id == "1918020023"]) # the actual class: seal
pred$pred_class[pred$id == "1918020023"] # the predicted class in each frame
target_acc_per_target_mis["1918020023", 2] # the accuracy: only 0.06

length(target.pix[["1918020023"]])
par(mfrow = c(5,4))
for (i in 1:17) {
  image <- image(target.pix.nopad[["1918020023"]][[i]])
}

### Another one 
first(pred$class[pred$id == "1809540102"])
pred$pred_class[pred$id == "1809540102"] 
target_acc_per_target_mis["1809540102", 2]

length(target.pix[["1809540102"]])
par(mfrow = c(4,4))
for (i in 1:16) {
  image <- image(target.pix.nopad[["1809540102"]][[i]])
}


# Actual: non-seal/ Predict: seal (0% acc)
target_acc_per_target_mis["1928490038", 2]
target_acc_per_target_mis["1903580027", 2] 

pred$pred_class[pred$id == "1903580027"]
target.var$valid[target.var$id == "1903580027"]

length(target.pix[["1903580027"]])
par(mfrow = c(4,4))
for (i in 1:15) {
  image <- image(target.pix.nopad[["1903580027"]][[i]])
}

## another one
pred$pred_class[pred$id == "1928490038"] 
target.var$valid[target.var$id == "1928490038"]

length(target.pix[["1928490038"]])
par(mfrow = c(5,4))
for (i in 1:19) {
  image <- image(target.pix.nopad[["1928490038"]][[i]])
}



# Save all results
save.image("C:/Users/iandurbach/Desktop/cnn_target.RData")


