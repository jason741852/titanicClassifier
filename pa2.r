require('lattice')
require('ggplot2')
require('methods')
require('caret') # confusionMatrix
require('ROCR') # ROC curve
require('e1071')   # SVM model
library('binr')
require('plyr')

normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }


df = read.csv("titanic3.csv", na.strings=c("", "NA"))
# this data set has 1309 rows

# this data set is missing 3855 values
missing_attribute_count <- sum(is.na(df))

# Split into train and test set
smp_size <- floor(0.80 * nrow(df))
set.seed(1)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train_data <- df[train_ind, ]
test_data <- df[-train_ind, ]

# Fill in the missing age data with mean of male and female ages in train and test set seperately
train_data$age[which(is.na(train_data$age) & train_data$sex=="female")] <- mean(train_data$age[which(train_data$sex=="female")], na.rm = TRUE)
train_data$age[which(is.na(train_data$age) & train_data$sex=="male")] <- mean(train_data$age[which(train_data$sex=="male")], na.rm = TRUE)
test_data$age[which(is.na(test_data$age) & test_data$sex=="female")] <- mean(test_data$age[which(test_data$sex=="female")], na.rm = TRUE)
test_data$age[which(is.na(test_data$age) & test_data$sex=="male")] <- mean(test_data$age[which(test_data$sex=="male")], na.rm = TRUE)

# Fill in missing embarked data with S, which is the most frequent value
train_data$embarked[which(is.na(train_data$embarked))] <- "S"
test_data$embarked[which(is.na(test_data$embarked))] <- "S"

# Only keep pclass, age, sex, sibsp, parch, embarked
train_data <- train_data[ -c(3, 8, 9, 10, 12, 13, 14) ]
test_data <- test_data[ -c(3, 8, 9, 10, 12, 13, 14) ]

hist(train_data$age, train_data$survived, breaks = seq(0, 100, by = 5))

# Check skewness and Kurtois of the data
print(c("age skewness: ", skewness(train_data$age)))
print(c("age kurtosis: ", kurtosis(train_data$age)))
train_data$age <- as.numeric(train_data$age)
test_data$age <- as.numeric(test_data$age)


# Bin Age and smooth by median
train_data["agebin"] <- NA
test_data["agebin"] <- NA
bins = c(0.092,8.15,16.1,24.1,32.1,40.1,48.1,56.1,64,72,80.1)
binss = c(0.092,4.16,8.15,12.1,16.1,20.1,24.1,28.1,32.1,36.1,40.1,44.1,48.1,52.1,56.1,60,64,68,72,76,80.1)

train_data$agebin <- .bincode(train_data$age,binss, TRUE,TRUE)
test_data$agebin <- .bincode(test_data$age,binss, TRUE,TRUE)


for(i in 1:20){
  train_data$age[which(train_data$agebin==i)]<- median(train_data$age[which(train_data$agebin==i)], na.rm = TRUE)
  test_data$age[which(test_data$agebin==i)]<- median(test_data$age[which(test_data$agebin==i)], na.rm = TRUE)

}

# Normalize binned data
dfNormage = as.data.frame(lapply(train_data["age"], normalize))
train_data["age"] <- dfNormage
dfNormage = as.data.frame(lapply(test_data["age"], normalize))
test_data["age"] <- dfNormage

# Check skewness and Kurtois of the data
print(c("age skewness after : ", skewness(train_data$age)))
print(c("age kurtosis after : ", kurtosis(train_data$age)))
#hist(df$age, df$survived, breaks = seq(0, 100, by = 5))




model <- glm(survived ~.,family=binomial(link='logit'),data=train_data)

summary(model)

prediction <- predict(model, test_data, type="response")

# Draw the decision boundary at 0.5 and assign the labels accordingly
prediction_label <- ifelse(prediction >= 0.5, 1, 0)

confMatrix <- confusionMatrix(prediction_label, test_data$survived, dnn=c("Prediction", "Reference"))
confMatrix

# Build ROC curve and AUC and find the best probability threshold
pred = prediction(prediction, test_data$survived)
roc = performance(pred, "tpr", "fpr")

plot(roc, lwd=2, colorize=TRUE)
title(main="ROC Curve of Logistic Regression Model")
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

auc = performance(pred, "auc")
auc = unlist(auc@y.values)
print(c(area_under_curve=auc))

acc.perf = performance(pred, measure = "acc")
plot(acc.perf)
title(main="AUC of Logistic Regression Model vs different prob_threshold")


ind = which.max(slot(acc.perf, "y.values")[[1]])
acc = slot(acc.perf, "y.values")[[1]][ind]
prob_threshold = slot(acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, prob_threshold = prob_threshold))



# ------------------- Learn SVM models -------------------

# Root mean square error
rmse <- function(error)
{
  sqrt(mean(error^2))
}



SVMLinearModel <- svm(survived ~ ., data = train_data, kernel = "linear", cost=1, epsilon=0.4)
SVMLinearPrediction <- predict(SVMLinearModel, test_data, type="response")

error <- test_data$survived - SVMLinearPrediction
SVMLinearPredictionRMSE <- rmse(error)
print(c(RMSE_of_Linear_kernel_SVM_before_tuning = SVMLinearPredictionRMSE))

SVMRadialModel <- svm(survived ~ ., data = train_data, kernel = "radial", cost=8, gamma=0.0625)
SVMRadialPrediction <- predict(SVMRadialModel, test_data, type="response")

error <- test_data$survived - SVMRadialPrediction
SVMRadialPredictionRMSE <- rmse(error)
print(c(RMSE_of_Radial_kernel_SVM_before_tuning = SVMRadialPredictionRMSE))

# Tuning
# tunedLinearSVM <- tune (svm, survived ~ .,  data = train_data,
#               ranges = list(epsilon = seq(0,1,0.05), cost = 2^(-2:9))
# )
# print(tunedLinearSVM)

# tunedRadialSVM <- tune.svm(survived ~ .,  data = train_data,
#               cost = 2^(2:9), kernel = "radial", gamma = 2^(-10:2))
# print(tunedRadialSVM)

# Confusion Matrix
SVMLinearPrediction_label <- ifelse(SVMLinearPrediction >= 0.5241331, 1, 0)
SVMRadialPrediction_label <- ifelse(SVMRadialPrediction >= 0.5511050, 1, 0)

confMatrixLSVM <- confusionMatrix(SVMLinearPrediction_label, test_data$survived, dnn=c("Prediction", "Reference"))
confMatrixLSVM

# Build ROC curve and AUC and find the best probability threshold for Linear SVM
predLSVM = prediction(SVMLinearPrediction, test_data$survived)
rocLSVM = performance(predLSVM, "tpr", "fxpr")

plot(rocLSVM, lwd=2, colorize=TRUE)
title(main="ROC Curve of SVM with Linear Kernel")
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

aucLSVM = performance(predLSVM, "auc")
aucLSVM = unlist(aucLSVM@y.values)
print(c(area_under_curve=aucLSVM))

accLSVM.perf = performance(predLSVM, measure = "acc")
plot(accLSVM.perf)
title(main="AUC of SVM with Linear Kernel vs different prob_threshold")


indLSVM = which.max(slot(accLSVM.perf, "y.values")[[1]])
accLSVM = slot(accLSVM.perf, "y.values")[[1]][indLSVM]
prob_thresholdLSVM = slot(acc.perf, "x.values")[[1]][indLSVM]
print(c(accuracy= accLSVM, prob_threshold = prob_thresholdLSVM))


confMatrixRSVM <- confusionMatrix(SVMRadialPrediction_label, test_data$survived, dnn=c("Prediction", "Reference"))
confMatrixRSVM

# Build ROC curve and AUC and find the best probability threshold for Radial SVM
predRSVM = prediction(SVMRadialPrediction, test_data$survived)
rocRSVM = performance(predRSVM, "tpr", "fpr")

plot(rocRSVM, lwd=2, colorize=TRUE)
title(main="ROC Curve of SVM with Radial Kernel")
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

aucRSVM = performance(predRSVM, "auc")
aucRSVM = unlist(aucRSVM@y.values)
print(c(area_under_curve=aucRSVM))

accRSVM.perf = performance(predRSVM, measure = "acc")
plot(accRSVM.perf)
title(main="AUC of SVM with Radial Kernel vs different prob_threshold")


indRSVM = which.max(slot(accRSVM.perf, "y.values")[[1]])
accRSVM = slot(accRSVM.perf, "y.values")[[1]][indRSVM]
prob_thresholdRSVM = slot(acc.perf, "x.values")[[1]][indRSVM]
print(c(accuracy= accRSVM, prob_threshold = prob_thresholdRSVM))
