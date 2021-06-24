filenames <- list.files("/home/apooreapo/Documents/Python Projects/diplomaThesis/starting_project/extracted_features/ultra_short_filtered/balanced", pattern="*.csv", full.names=TRUE)
# file_list <- list.files(path="/home/apooreapo/Documents/R projects/diplomaThesis/data_for_machine_learning/ultra_short_filtered_healthy_extra.csv")
for (i in 1:length(filenames)){
#for (i in 1:1){
  
  # CALCULATE NAME OF FILE
  
  final_names = strsplit(filenames[i], "_")[[1]]
  final_name = final_names[length(final_names)]
  print(final_name)
  
  # READ CSV FILES
  
  data_current <- read.csv(file = filenames[i])
  summary(data_current)
  #c1 = c(2:21,24:26) # use in any case except for CAD (Training/testing)
  c1 = c(4:23,26:28) # use only in CAD testcase
  newDataCurrent <-data_current[,c1]
  newDataCurent = newDataCurrent[complete.cases(newDataCurrent),]
  n = nrow(newDataCurrent)
  
  # PCA HERE
  
  newDataCurrent.pca <- predict(train.pca, newdata = newDataCurrent)
  newDataCurrentMat <- as.data.frame(newDataCurrent.pca[,1:13])
  
  # MAKE PREDICTION HERE

  predicted = predict(svm_model005, newDataCurrentMat)
  counted_ones = length(which(predicted == 1))
  counted_zeros = length(which(predicted == 0))
  print(counted_ones + counted_zeros)
  print(counted_ones / (counted_ones + counted_zeros))
  
  # WRITE TO CSV
  
  final_name = paste("/home/apooreapo/Documents/R projects/diplomaThesis/results/training_testing/res_", final_name)
  write.csv(predicted,file = final_name)
}