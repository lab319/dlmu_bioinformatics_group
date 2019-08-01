rm(list=ls())
library(glmnet) 
library(caret)
library(pROC)
library(ROCR)
library(e1071)
library(caTools)
x<-read.table("KIRP_mrna.txt",sep='\t',header=TRUE)
y<-read.table("KIRP_label.txt",sep='\t',header=TRUE)
dim(x)
dim(y)
head(x[1:5,1:5])

for(a in 1:9){
	print(0.1*a)
	set.seed(1)
	inTrain <- createDataPartition(y=y$stage, p=0.8, list=F) 
	x_train<- x[inTrain, ]
	x_test<-x[-inTrain, ]
	y_train<- y[inTrain, ]
	y_test<-y[-inTrain, ]
	dim(x_train)
	dim(x_test)
	length(y_train)
	length(y_test)

	x_train <- scale(x_train)
	x_test <- (x_test - matrix(attr(x_train, "scaled:center"), nrow = nrow(x_test), ncol = ncol(x_test), byrow = TRUE)) / matrix(attr(x_train, "scaled:scale"), nrow = nrow(x_test), ncol = ncol(x_test), byrow = TRUE)

	x_train<-as.matrix(x_train)
	y_train<-as.matrix(y_train)
	x_test<-as.matrix(x_test)
	y_test<-as.matrix(y_test)

	CV = cv.glmnet(x_train, y=y_train, family='binomial', type.measure="auc",nfolds=5, alpha=0.1*a, nlambda=100)
	#print(plot(CV))
	max_auc=max(CV$cvm)
	print(max_auc)
	print(CV$lambda.1se)
}

#alpha: 0.1
#max_auc: 0.8955454
#lambda: 1.707213


