file_cad_as_well = "/home/apooreapo/Documents/R projects/diplomaThesis/data_for_machine_learning/ultra_short_filtered_cad_as_well.csv"
data_cad_as_well <- read.csv(file = file_cad_as_well)
c33 = c(3:22, 25:27)
data_cad_as_well <- data_cad_as_well[,c33]
data_cad_as_well = data_cad_as_well[complete.cases(data_cad_as_well),]
cad_as_well.pca <- predict(train.pca, newdata = data_cad_as_well)
cadaswellmat <- as.data.frame(cad_as_well.pca[,1:12])
datafile <- read.csv(file = file_cad_as_well)
datafile = datafile[complete.cases(datafile),]
fileTitle = datafile$File.title
predcadaswell = prediction(svm_model005, cadaswellmat)


i <- 0
nm = unique(fileTitle)
dict_correct = rep(0,length(nm))
names(dict_correct) = nm
dict_full = dict_correct
dict_ratio = dict_correct
for (prediction_name in fileTitle){
  i = i + 1
  for (name in names(dict_correct)){
    if (prediction_name == name){
      dict_full[name] = dict_full[name] + 1
      if (predcadaswell[i] == 1){
        dict_correct[name] = dict_correct[name] + 1
      }
      break
    }
  }
}
for (name in names(dict_correct)){
  dict_ratio[name] = dict_correct[name] / dict_full[name]
}






file_no_cad_other_d = "/home/apooreapo/Documents/R projects/diplomaThesis/data_for_machine_learning/ultra_short_filtered_no_cad_other_diseases.csv"
data_no_cad_other_d <- read.csv(file = file_no_cad_other_d)
c33 = c(3:22, 25:27)
data_no_cad_other_d <- data_no_cad_other_d[,c33]
data_no_cad_other_d = data_no_cad_other_d[complete.cases(data_no_cad_other_d),]
no_cad_other_d.pca <- predict(train.pca, newdata = data_no_cad_other_d)
nocadotherdmat <- as.data.frame(no_cad_other_d.pca[,1:12])
datafile <- read.csv(file = file_no_cad_other_d)
datafile = datafile[complete.cases(datafile),]
fileTitleNOD = datafile$File.title
prednocadotherd = prediction(svm_model005, nocadotherdmat)
rm(datafile)


i <- 0
nmnod = unique(fileTitleNOD)
dict_correct_nod = rep(0,length(nmnod))
names(dict_correct_nod) = nmnod
dict_full_nod = dict_correct_nod
dict_ratio_nod = dict_correct_nod
for (prediction_name in fileTitleNOD){
  i = i + 1
  for (name in names(dict_correct_nod)){
    if (prediction_name == name){
      dict_full_nod[name] = dict_full_nod[name] + 1
      if (prednocadotherd[i] == 1){
        dict_correct_nod[name] = dict_correct_nod[name] + 1
      }
      break
    }
  }
}
for (name in names(dict_correct_nod)){
  dict_ratio_nod[name] = dict_correct_nod[name] / dict_full_nod[name]
}