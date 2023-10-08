f<- read.csv("Placement_Data_Full_Class.csv")
f <- f[,-15]
f <- f[,-1]
f$gender = as.factor(f$gender)
f$ssc_p = as.factor(f$ssc_p)
f$ssc_b = as.factor(f$ssc_b)
f$hsc_s = as.factor(f$hsc_s)
f$hsc_b = as.factor(f$hsc_b)
f$degree_t = as.factor(f$degree_t)
f$workex = as.factor(f$workex)
f$specialisation = as.factor(f$specialisation)
f$status = as.factor(f$status)
library("ROSE")
f <- ovun.sample(status~., data = f, method = "both", p = 0.5, seed = 222)$data
set.seed(123)
a <- sample(2, nrow(f), replace = TRUE, prob = c(0.75, 0.25))
traning <- f[a==1,]
test <- f[a==2,]
library(nnet)
model <- multinom(status~., data = traning)
library("caret")
p1 <- predict(model, traning)
confusionMatrix(p1, traning$status)
