setwd("F:/GONGFANGZHE/!哲小葵is programming/哲小葵studying RRRRRRRRRRRRRR/Practical Machine Learning - Coursera")
#Preparation work
library(caret)
library(rattle)

#1. Getting, Cleaning and Exploring the data
TrainData <- read.csv('./pml-training.csv', header=T)
TestData <- read.csv('./pml-testing.csv', header=T)
dim(TrainData)
dim(TestData)
str(TrainData)

#Cleaning the input data 
# Here we get the indexes of the columns having at least 90% of NA 
#or blank values on the training dataset
indColToRemove <- which(colSums(is.na(TrainData) |TrainData=="")>0.9*dim(TrainData)[1]) 
TrainDataClean <- TrainData[,-indColToRemove]
TrainDataClean <- TrainDataClean[,-c(1:7)]
dim(TrainDataClean)
# Do the same for the test set
indColToRemove <- which(colSums(is.na(TestData) |TestData=="")>0.9*dim(TestData)[1]) 
TestDataClean <- TestData[,-indColToRemove]
TestDataClean <- TestDataClean[,-1]
dim(TestDataClean)
str(TestDataClean)

#2. Preparing the datasets for prediction
set.seed(12345)
inTrain1 <- createDataPartition(TrainDataClean$classe, p=0.75, list=FALSE)
Train1 <- TrainDataClean[inTrain1,]
Test1 <- TrainDataClean[-inTrain1,]
dim(Train1)
dim(Test1)

#3.Model building
#3.1：Train with classification tree
trControl <- trainControl(method="cv", number=5)
model_CT <- train(classe~., data=Train1, method="rpart", trControl=trControl)
#print(model_CT)
fancyRpartPlot(model_CT$finalModel)

trainpred <- predict(model_CT,newdata=Test1)
confMatCT <- confusionMatrix(Test1$classe,trainpred)

# display confusion matrix and model accuracy
confMatCT$table
confMatCT$overall[1]

#3.2：Train with random forests
model_RF <- train(classe~., data=Train1, method="rf", trControl=trControl, verbose=FALSE)
print(model_RF)

plot(model_RF,main="Accuracy of Random forest model by number of predictors")
trainpred <- predict(model_RF,newdata=Test1)
confMatRF <- confusionMatrix(Test1$classe,trainpred)
# display confusion matrix and model accuracy
confMatRF$table
confMatRF$overall[1]
names(model_RF$finalModel)
model_RF$finalModel$classes
plot(model_RF$finalModel,main="Model error of Random forest model by number of trees")
# Compute the variable importance 
MostImpVars <- varImp(model_RF)
MostImpVars

#3.3: Train with gradient boosting method
model_GBM <- train(classe~., data=Train1, method="gbm", trControl=trControl, verbose=FALSE)
print(model_GBM)
plot(model_GBM)
trainpred <- predict(model_GBM,newdata=Test1)

confMatGBM <- confusionMatrix(Test1$classe,trainpred)
confMatGBM$table
confMatGBM$overall[1]

FinalTestPred <- predict(model_RF,newdata=TestDataClean)
FinalTestPred

