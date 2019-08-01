rm(list=ls())
library(glmnet) 
library(caret)
library(pROC)
library(ROCR)
library(e1071)
library(caTools)
x<-read.table("KIRP_methy.txt",sep='\t',header=TRUE)
y<-read.table("KIRP_label.txt",sep='\t',header=TRUE)
dim(x)
dim(y)
head(x[1:5,1:5])

sum_AUC<-0
sum_ACC<-0
sum_PR_AUC<-0
sum_MCC<-0
sum_Sen<-0
sum_Spe<-0

for(i in 0:99)
  {	
	##############MCC############################
	Mcc <- function (act, pred){
  	TP <- sum(act == 1 & pred == 1)
  	TN <- sum(act == 0 & pred == 0)
  	FP <- sum(act == 0 & pred == 1)
  	FN <- sum(act == 1 & pred == 0)

  	denom <- as.double(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
  	if (any((TP+FP) == 0, (TP+FN) == 0, (TN+FP) == 0, (TN+FN) == 0)) denom <- 1
  	mcc <- ((TP*TN)-(FP*FN)) / sqrt(denom)
  	return(mcc)
	}

	#####################pr_auc#######################
	pr_auc <- function(obs, pred) {
  	xx.df <- prediction(pred, obs)
  	perf <- performance(xx.df, "prec", "rec")
  	xy <- data.frame(recall=perf@x.values[[1]], precision=perf@y.values[[1]])
  
  	xy <- subset(xy, !is.nan(xy$precision))

  	xy <- rbind(c(0, 0), xy)
  
  	res <- trapz(xy$recall, xy$precision)
  	res
	}


	print(i)
	set.seed(i)
	inTrain <- createDataPartition(y=y$stage, p=0.8, list=F) 
	x_train<- x[inTrain, ]
	x_test<-x[-inTrain, ]
	y_train<- y[inTrain, ]
	y_test<-y[-inTrain, ]

	x_train <- scale(x_train)
	x_test <- (x_test - matrix(attr(x_train, "scaled:center"), nrow = nrow(x_test), ncol = ncol(x_test), byrow = TRUE)) / matrix(attr(x_train, "scaled:scale"), nrow = nrow(x_test), ncol = ncol(x_test), byrow = TRUE)

	x_train<-as.matrix(x_train)
	y_train<-as.matrix(y_train)
	x_test<-as.matrix(x_test)
	y_test<-as.matrix(y_test)


	fit = glmnet(x_train, y=y_train, family='binomial', alpha=0.1, lambda= 0.3935762)
	print(fit)
	
	predict<-predict(fit,x_test,type="class",s=0.3935762)
	predictions<-predict.glmnet(fit,x_test,type="class",s=0.3935762)
	predictions

	pred <- prediction(predictions, y_test)
	perf <- performance(pred,"tpr","fpr")

	auc <- performance(pred, "auc")@y.values    
	result<-confusionMatrix(factor(predict, levels = 1:0),factor(y_test, levels = 1:0))
	Acc<-result$overall['Accuracy']
	Sen<-result$byClass['Sensitivity']
	Spe<-result$byClass['Specificity']
	Mcc<-Mcc(y_test,predict)
	pr_auc<-pr_auc(y_test,predictions)
    
	sum_AUC = sum_AUC + auc[[1]]
	sum_ACC = sum_ACC + Acc
	sum_PR_AUC =sum_PR_AUC + pr_auc
	sum_MCC = sum_MCC + Mcc
	sum_Sen = sum_Sen + Sen
	sum_Spe = sum_Spe + Spe

	print(auc)
	print(Acc)
	print(pr_auc)
	print(Mcc)
	print(Sen)
	print(Spe)

}

cat("AUC_mean:",round(sum_AUC/100,3),"\n")
cat("ACC_mean:",round(sum_ACC/100,3),"\n")
cat("PRAUC_mean:",round(sum_PR_AUC/100,3),"\n")
cat("MCC_mean:",round(sum_MCC/100,3),"\n")
cat("Sen_mean:",round(sum_Sen/100,3),"\n")
cat("Spe_mean:",round(sum_Spe/100,3),"\n")


#AUC_mean: 0.845 
#ACC_mean: 0.842 
#PRAUC_mean: 0.918 
#MCC_mean: 0.524 
#Sen_mean: 0.961 
#Spe_mean: 0.465 


 











