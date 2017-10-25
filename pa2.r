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

# Fill in the missing age data with mean of male and female ages
df$age[which(is.na(df$age) & df$sex=="female")] <- mean(df$age[which(df$sex=="female")], na.rm = TRUE)
df$age[which(is.na(df$age) & df$sex=="male")] <- mean(df$age[which(df$sex=="male")], na.rm = TRUE)

# Fill in missing embarked data with S, which is the most frequent value
df$embarked[which(is.na(df$embarked))] <- "S"

# Only keep pclass, age, sex, sibsp, parch, embarked
df <- df[ -c(3, 8, 9, 10, 12, 13, 14) ]
hist(df$age, df$survived, breaks = seq(0, 100, by = 5))

# Check skewness and Kurtois of the data
print(c("age skewness: ", skewness(df$age)))
print(c("age kurtosis: ", kurtosis(df$age)))
df$age <- as.numeric(df$age)
# dfNormage = as.data.frame(lapply(df["age"], normalize))

standardizedAge <- scale(df['age'])
df$age <- standardizedAge

# df["agebin"] <- NA
# bins = c(0.092,8.15,16.1,24.1,32.1,40.1,48.1,56.1,64,72,80.1)
# binss = c(0.092,4.16,8.15,12.1,16.1,20.1,24.1,28.1,32.1,36.1,40.1,44.1,48.1,52.1,56.1,60,64,68,72,76,80.1)
#
# df$agebin <- .bincode(df$age,binss, TRUE,TRUE)
#
# for(i in 1:20){
#   df$age[which(df$agebin==i)]<- median(df$age[which(df$agebin==i)], na.rm = TRUE)
# }


# Check skewness and Kurtois of the data
print(c("age skewness after : ", skewness(df$age)))
print(c("age kurtosis after : ", kurtosis(df$age)))
#hist(df$age, df$survived, breaks = seq(0, 100, by = 5))



# Split into train and test set
smp_size <- floor(0.80 * nrow(df))
set.seed(1)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train_data <- df[train_ind, ]
test_data <- df[-train_ind, ]

model <- glm(survived ~.,family=binomial(link='logit'),data=train_data)

#summary(model)

prediction <- predict(model, test_data, type="response")

# Draw the decision boundary at 0.5 and assign the labels accordingly
prediction_label <- ifelse(prediction >= 0.5, 1, 0)


confMatrix <- confusionMatrix(prediction_label, test_data$survived, dnn=c("Prediction", "Reference"))
confMatrix

# Build ROC curve and AUC and find the best probability threshold
pred = prediction(prediction, test_data$survived)
roc = performance(pred, "tpr", "fpr")

plot(roc, lwd=2, colorize=TRUE)
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

auc = performance(pred, "auc")
auc = unlist(auc@y.values)
print(c(area_under_curve=auc))

acc.perf = performance(pred, measure = "acc")
plot(acc.perf)

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

SVMRadialModel <- svm(survived ~ ., data = train_data, kernel = "radial")
SVMRadialPrediction <- predict(SVMRadialModel, test_data, type="response")

error <- test_data$survived - SVMRadialPrediction
SVMRadialPredictionRMSE <- rmse(error)
print(c(RMSE_of_Radial_kernel_SVM_before_tuning = SVMRadialPredictionRMSE))

# tuning
# tunedLinearSVM <- tune(svm, survived ~ .,  data = train_data,
#               ranges = list(epsilon = seq(0,1,0.05), cost = 2^(-2:9))
# )
# print(tunedLinearSVM)

# tunedRadialSVM <- tune.svm(survived ~ .,  data = train_data,
#               cost = 2^(2:9), kernel = "radial"
# )
# print(tunedRadialSVM)
