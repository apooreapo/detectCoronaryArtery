library(e1071)
library(MLmetrics)
library(class)

file_data = "/home/apooreapo/Documents/R projects/diplomaThesis/data_for_machine_learning/ultra_short_filtered_full.csv"
data <- read.csv(file = file_data)
c33 = c(2:26)
newData2 <- data[,c33]
newData2 = newData2[complete.cases(newData2),]
#cad_as_well.pca <- predict(train.pca, newdata = data_cad_as_well)
#cadaswellmat <- as.data.frame(cad_as_well.pca[,1:12])
#datafile <- read.csv(file = file_cad_as_well)
#datafile = datafile[complete.cases(datafile),]
fileTitle = newData2$File.title
nm = unique(fileTitle)
nm2 = levels(nm)
tt_big = c()
ratios_big = c()
#predcadaswell = prediction(svm_model005, cadaswellmat)
for (names in nm2){
  if (names != ""){
    ck = c(1:20,23:25)
    to_be_checked = newData2[newData2$File.title == names,]
    to_be_checked = to_be_checked[,ck]
    nfull = nrow(to_be_checked)
    rest = newData2[newData2$File.title != names,]
    rest_cad = rest[rest$Has.CAD == 1,]
    rest_no_cad = rest[rest$Has.CAD == 0,]
    ncad = nrow(rest_cad)
    nnocad = nrow(rest_no_cad)
    no_cad_train_index = sample(1:nnocad, size = 40000, replace = FALSE)
    cad_train_index = sample(1:ncad, size = 40000, replace = FALSE)
    rest_train_cad = rest_cad[cad_train_index,]
    rest_train_no_cad = rest_no_cad[no_cad_train_index,]
    rest_train = rbind(rest_train_cad, rest_train_no_cad)
    rest_train_res = rest_train[,22]
    
    rest_train = rest_train[,ck]
    to_be_checked.pca <- predict(train.pca, newdata = to_be_checked)
    to_be_checked_mat <- as.data.frame(to_be_checked.pca[,1:12])
    rest_train.pca <- predict(train.pca, newdata = rest_train)
    rest_train_mat <- as.data.frame(rest_train.pca[,1:12])
    rest_train_mat$Has.CAD = rest_train_res
    
    print("Training model")
    svm_model = svm(Has.CAD ~ ., kernel="radial", type="C-classification", data = rest_train_mat, gamma = 0.001)
    print("Model trained, predicting values")
    pred = predict(svm_model, to_be_checked_mat)
    print("Values predicted")
    positive = length(pred[pred == 1])
    print(positive)
    ratio = positive / nfull
    print(ratio)
    print(names)
    
    ratios_big = c(ratios_big, ratio)
    tt_big = c(tt_big, names)
  }
}

# tt is for gamma=0.001, 4000
# tt1 is for gamma=0.1, 4000
# tt_big is for gamma=0.001, 40000