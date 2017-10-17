require('lattice')
require('ggplot2')
require('methods')
require('caret') # confusionMatrix
require('ROCR') # ROC curve
require('e1071')   # SVM model



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



#is.na(mydata)


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



SVMLinearModel <- svm(survived ~ ., data = train_data, kernel = "linear")
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
#               ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
# )
# print(tunedLinearSVM)

tunedRadialSVM <- tune.svm(survived ~ .,  data = train_data,
              cost = 2^(2:9), kernel = "radial"
)
print(tunedRadialSVM)
