rm(list=ls())
library(e1071)
library(MLmetrics)
library(class)
file1 = "/home/apooreapo/Documents/R projects/diplomaThesis/data_for_machine_learning/ultra_short_filtered_full.csv"
data <- read.csv(file = file1)
c1 = c(2:21,23:26)
newData <-data[,c1]
newData = newData[complete.cases(newData),]
n = nrow(newData)
random_selection <- function(percentage) {
  trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
  train <<- newData[trainIndex,]
  test <<- newData[-trainIndex,]
}

trainIndex = sample(1:n, size = round(0.1*n), replace=FALSE)
train = newData[trainIndex,]
test = newData[-trainIndex,]
n2 = nrow(train)
trainIndex2 = sample(1:n2, size = round(0.7*n2), replace=FALSE)
train2 = train[trainIndex2,]
test2 = train[-trainIndex2,]
print("hi")

# PCA HERE #
l1 = length(newData)
trainres = train[,21]
trainmod = train[,-21]
testres = test[,21]
testmod = test[,-21]
train.pca <- prcomp(trainmod, center = TRUE, scale = TRUE)
trainmat <- as.data.frame(train.pca$x[,1:16])
test.pca <- predict(train.pca, newdata = testmod)
testmat <- as.data.frame(test.pca[,1:16])
# PCA dimension 16 seems to be the best choice

# KNN

pred2 = knn(train = trainmat, test = testmat, cl = trainres, k = 7)


train2mat$Has.CAD = train2res

svm_model = svm(Has.CAD ~ ., kernel="radial", type="C-classification", data = train2mat, gamma = 0.1)
l1 = length(newData)
pred = predict(svm_model, test2mat)
#pred = predict(svm_model, test2[,c(1:(l1-4),(l1-2):l1)])
Accuracy(pred, test2[,c(l1-3)])
Recall(test2[,c(l1-3)], prediction)
Precision(test2[,c(l1-3)], prediction)
F1_Score(test2[,c(l1-3)], prediction)
training_error = c()
testing_error = c()
gammavalues = c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000)
for (gamma in gammavalues) {
  svm_model = svm(Has.CAD ~ ., kernel="radial", type="C-classification", 
                  data = train, gamma = gamma)
  pred = predict(svm_model, test[,c(1:(l1-5),(l1-3):l1)])
  acc = Accuracy(pred, test[,c(l1-4)])
  testing_error=c(testing_error,1-acc)
  
  pred = predict(svm_model, train[,c(1:(l1-5),(l1-3):l1)])
  acc = Accuracy(pred, train[,c(l1-4)])
  training_error=c(training_error,1-acc)
  
  print(acc)
  
}
plot(training_error, type = "l", col="blue", ylim = c(0, 0.5), xlab
     = "Gamma", ylab = "Error", xaxt = "n")
axis(1, at = 1:length(gammavalues), labels = gammavalues)
lines(testing_error, col="red")
# best value seems to be 0.1 or something near there
# results: testing Acc: 96.7%, training Acc: 98.1%

# uncomment below to get a different type of evaluation
#test = newData[data[,29]=="16265.csv" | 
#                data[,29]=="s20321.csv" | 
#data[,29]=="s20191.csv" | data[,29]=="16272.csv",]
#test = newData[data[,29]=="16265.csv" |
#                data[,29]=="s20321.csv" |
#              data[,29]=="s20191.csv" | data[,29]=="16272.csv",]

