
# CNN-RNN Prediction 
library(tidyverse)
library(ggplot2)

# Merge the original validation data and predicted class  
pred_class <- history %>% predict(x_valid) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
valid <- cbind(valid, pred_class)

# Tune threshold
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
    mutate(targ_class = ifelse(meanseal > i, 1, 0))
  valid_table[N,,] <- table(Pred = v$targ_class, Actual = v$true_class)
  valid_acc[N] <- round((valid_table[N,1,1] + valid_table[N,2,2]) / sum(valid_table[N,,]) * 100, 2)
}

v <- cbind(thre, valid_acc) %>% as.data.frame()
best_valid_thre <- thre[which.max(valid_acc)]
same_acc <- data.frame(thre = thre[valid_acc == max(valid_acc)], acc = valid_acc[valid_acc == max(valid_acc)])
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

# The test result (accuracy & confusion matrix) under the best threshold
valid_result <- valid %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))
valid_table <- table(Pred = valid_result$targ_class, Actual = valid_result$true_class)
valid_acc <- round((valid_table[1,1] + valid_table[2,2]) / sum(valid_table) * 100, 2)


# Train data
pred_class <- history %>% predict(x_train) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
train <- cbind(train, pred_class) 
#train <- train %>% select(-.)

train_result <- train %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0))

train_table<- table(Pred = train_result$targ_class, Actual = train_result$true_class)
train_acc <- round((train_table[1,1] + train_table[2,2]) / sum(train_table) * 100, 2)



# Test data
pred_class <-  history %>% predict(x_test) %>% k_argmax() %>% array() %>% as.data.frame()
colnames(pred_class) <- c("pred_class")
test <- cbind(test, pred_class)
test_result <- test %>% 
  group_by(id) %>% 
  summarize(meanseal = mean(pred_class), true_class = first(class)) %>%
  mutate(targ_class = ifelse(meanseal > best_valid_thre, 1, 0)) 

test_table <- table(Pred = test_result$targ_class, Actual = test_result$true_class)
test_acc <- round(((test_table[1,1] + test_table[2,2]) / sum(test_table)) * 100, 2)

train_acc
valid_acc
test_acc
rnn_train_targ_acc <- train_acc
rnn_test_targ_acc <- test_acc
rnn_valid_targ_acc <- valid_acc
rnn_train_targ_acc; rnn_test_targ_acc; rnn_valid_targ_acc


# Frame-level accuracy------------------------------
# Merge train & test & validation dataset
train$dataset <- c("train")
valid$dataset <- c("valid")
test$dataset <- c("test")
rnn_pred <- rbind(train, valid, test)

test_table_frame <- table(Pred = test$class, Actual = test$pred_class)
valid_table_frame <- table(Pred = valid$class, Actual = valid$pred_class)
train_table_frame <- table(Pred = train$class, Actual = train$pred_class)

test_table_frame; valid_table_frame; train_table_frame
round((valid_table_frame[1,1] + valid_table_frame[2,2]) / sum(valid_table_frame) * 100, 2)

# false positive rate
round(test_table_frame[2,1] / sum(test_table_frame) * 100, 2)
# false negative rate
round(test_table_frame[1,2] / sum(test_table_frame) * 100, 2)





rnn_train_frame_acc; rnn_test_frame_acc; rnn_valid_frame_acc

# Compare the accuracy at the target level and frame level
(rnn_train_targ_acc - rnn_train_frame_acc) / rnn_train_frame_acc * 100
(rnn_test_targ_acc - rnn_test_frame_acc) / rnn_test_frame_acc * 100
(rnn_valid_targ_acc - rnn_valid_frame_acc) / rnn_valid_frame_acc * 100


rnn_pred <- rnn_pred[rnn_pred$dataset == "test",]
# Accuracy per target
correct_class_per_tagret <- NULL
rnn_target_acc_per_target <- NULL

for (id in c(unique(rnn_pred$id))) {
  correct_class_per_tagret[id] <- nrow(rnn_pred[rnn_pred$id == id & rnn_pred$class == rnn_pred$pred_class,])
  rnn_target_acc_per_target[id] <- round(correct_class_per_tagret[id] / nrow(rnn_pred[rnn_pred$id == id,]), 2)
}

# Keep the accuracy and add id
rnn_target_acc_per_target <- rnn_target_acc_per_target %>% unlist() %>% as.data.frame()
colnames(rnn_target_acc_per_target) <- c("rnn_target_acc")
rnn_target_acc_per_target <- cbind(id = unique(rnn_pred$id), rnn_target_acc_per_target)

# The distribution of frame accuracy 
head(rnn_target_acc_per_target)
ggplot(rnn_target_acc_per_target, aes(rnn_target_acc * 100)) +
  geom_histogram(bins = 30, fill = "red", alpha = .6) +
 # scale_y_continuous(limits = c(0,60)) +
  theme_bw() +
  xlab("accuracy per target (%)") 


# acc = 0
nrow(rnn_target_acc_per_target[rnn_target_acc_per_target$target_acc == 0,])


# Check the misclassified target 
rnn_target_acc_per_target_mis <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc != 1,]
rnn_target_acc_per_target_mis
# false positive rate (nonseal-seal)
nrow(rnn_target_acc_per_target_mis[target.var$valid[target.var$id == rnn_target_acc_per_target_mis$id] == "Seal",])
# false negative (1-0)
nrow(rnn_target_acc_per_target_mis[target.var$valid[target.var$id == rnn_target_acc_per_target_mis$id] != "Seal",])



# -------------------------------------------------
# Let's have a look at which targets are prefect classified by CNN!

# Targets with 100% accuracy
rnn_acc_100 <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc == 1,]
rnn_acc_100

# the proportion of seals accounts for 100% accuracy
nrow(rnn_acc_100[rnn_pred$id == rnn_acc_100$id & rnn_pred$valid == "Seal",]) 


# Check the target with lower accuracy (e.g. <0.5)
rnn_target_acc_per_target_mis <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc < 100,]
nrow(rnn_target_acc_per_target_mis)

rnn_target_acc_per_target_mis <- rnn_target_acc_per_target[rnn_target_acc_per_target$rnn_target_acc == 0,]

# Type one/two error
first(rnn_pred$class[rnn_pred$id == rnn_target_acc_per_target_mis$id])
rnn_pred$pred_class[rnn_pred$id == rnn_target_acc_per_target_mis$id] 
## 1-0/Seal-No Seal: Type One error
6

# Change figure margins
par(mar = c(1,1,1,1))

## seal/non-seal (acc: 0)
first(rnn_pred$class[rnn_pred$id == "1809500048"]) 
rnn_pred$pred_class[rnn_pred$id == "1809500048"]
length(target.pix[["1809500048"]])
par(mfrow = c(4,8))
for (i in 1:31) {
  image <- image(target.pix.nopad[["1809500048"]][[i]])
}

first(rnn_pred$class[rnn_pred$id == "1836210026"]) 
rnn_pred$pred_class[rnn_pred$id == "1836210026"]
length(target.pix[["1836210026"]])
par(mfrow = c(5,4))
for (i in 1:20) {
  image <- image(target.pix.nopad[["1836210026"]][[i]])
}


first(rnn_pred$class[rnn_pred$id == "1735000011"]) 
rnn_pred$pred_class[rnn_pred$id == "1735000011"]
length(target.pix[["1735000011"]])
par(mfrow = c(4,4))
for (i in 1:13) {
  image <- image(target.pix.nopad[["1735000011"]][[i]])
}


