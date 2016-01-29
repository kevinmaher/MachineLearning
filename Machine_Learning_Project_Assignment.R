## Machine Learning Project Assignment


## Load the required libraries
library(caret)
library(gbm)
library(randomForest)
library(plyr)
library(klaR)
library(MASS)

## Get the Data
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

##Cleaning the data
##We can remove some of the variables that do not help predict the movement. 
##(This will also help speed up some of the modeling execution times later) "X", ##"user_name","raw_timestamp_part_1","raw_timestamp_part_2" ,"cvtd_timestamp"      

removecols <- grep("X|user_name|raw_timestamp_part_1|raw_timestamp_part_2|cvtd_timestamp", names(training))
training <- training[, -removecols]
testing <- testing[, -removecols]
dim(training)

## set classe as a factor variable to enable modelling 
training$classe = factor(training$classe)

##Removing zero covariates (ones near 0)
nearzero <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nearzero$nzv==FALSE]

## remove columns where all na values
training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]

dim(training); dim(testing)

##partitioning the training set into two
##We need to partition the training set into 2 so we can find  the minimal out-of-sample error, I'm going to use a 75/25 split.

splittrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
trainfirst <- training[splittrain, ]
traincrossval <- training[-splittrain, ]
dim(trainfirst)
dim(traincrossval)

## Comparing various models to find the best fit
##I decided to try and fit 3 different models.

##Set Seed to reproduce the work
set.seed(54321)

##Fit a Gradient Boosting Machine model
fitgbm = train(classe ~ .,data = trainfirst,method = "gbm", trControl=trainControl(method = "cv", number = 3))
gbm <- confusionMatrix(predict(fitgbm, traincrossval), traincrossval$classe)
gbm

##Fit a native bayes model
fitnbay <- train(classe ~ ., data = trainfirst, method = "nb", trControl=trainControl(method = "cv", number = 3))
nbay <- confusionMatrix(predict(fitnbay, traincrossval), traincrossval$classe)
nbay

##Fit a Random Forest model
fitrf = train(classe ~ .,data = trainfirst,method = "rf", trControl=trainControl(method = "cv", number = 3))
rf <- confusionMatrix(predict(fitrf, traincrossval), traincrossval$classe)
rf

##Compare accuracy of the three models
##From the table below you can see that the "Random Forest" model has the best accuracy with regard to this exercise.
acccuracytable <- data.frame(gbm$overall[1], nbay$overall[1], rf$overall[1])
acccuracytable

##The expected out-of-sample error is calculated as 1 - accuracy for predictions made 
##against the cross-validation set. In this example the best fitting model is the Random Forest model and the out of sample error = 0.002650897 which is 0.2%.

## Predict classe for test data
##use the best predictor (Random Forests) on the test data  to predict classe for each of the 20 samples
rftesting <- predict(fitrf, testing)
rftesting

