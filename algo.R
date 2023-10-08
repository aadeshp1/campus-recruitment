f<- read.csv("Placement_Data_Full_Class.csv")
f <- f[,-15]
f <- f[,-1]
f$gender = as.factor(f$gender)
f$ssc_b = as.factor(f$ssc_b)
f$hsc_s = as.factor(f$hsc_s)
f$hsc_b = as.factor(f$hsc_b)
f$degree_t = as.factor(f$degree_t)
f$workex = as.factor(f$workex)
f$specialisation = as.factor(f$specialisation)
f$status = as.factor(f$status)
library("caret")
library("ROSE")
f <- ovun.sample(status~., data = f, method = "both", p = 0.5, seed = 222)$data
set.seed(123)
a <- sample(2, nrow(f), replace = TRUE, prob = c(0.75, 0.25))
traning <- f[a==1,]
test <- f[a==2,]
#Logistic Regression----
cat("\n Logistic Regression \n")
library(nnet)
lr <- multinom(status~., data = traning)
p1 <- predict(lr, test)+
confusionMatrix(p1, test$status)
#Random Forest----
cat("\n Random Forest \n")
library(randomForest)
rf <- randomForest(status~., data = traning, ntree = 500)
f1 <- predict(rf, test)
confusionMatrix(f1, test$status)
#Support Vector Machine----
cat("\n Support Vector Machine \n")
library("e1071")
svm <- svm(status~.,data = traning)
k1 <- predict(svm, test)
confusionMatrix(k1, test$status)
