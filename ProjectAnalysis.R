print ("This File is created with in RStudio")
print ("And now it lives on GitHub")


ibrary(caretEnsemble)
# library(caTools)
library(caret)
library(PerformanceAnalytics)
library(dplyr)
library(jtools)
library(ROCR)
library(pROC)
library(randomForest)
library(glmnet)
library(gbm)
library(mlbench)
library(rpart)
library(ROSE)
install.packages("ROSE")

########################### START HERE ########################################
################################ Analyzing General Linear Model with ###########
##################################         Other Models               ##########
################################################################################
risk_estimate = read.table(file = "risk.txt", header = F)

colnames(risk_estimate) <- c("menopaus", "agegrp", "density", "race", "Hispanic",
                             "bmi", "agefirst", "nrelbc", "brstproc", "lastmamm",
                             "surgmeno", "hrt", "invasive", "cancer", "training", "count")
tail(risk_estimate)

############# PREPROCESSING DATA~~~~~~~~~~~~~~
risk_estimate$menopaus <-factor(risk_estimate$menopaus)
risk_estimate$agegrp <-factor(risk_estimate$agegrp)
risk_estimate$race <-factor(risk_estimate$race)
risk_estimate$nrelbc <-factor(risk_estimate$nrelbc)
risk_estimate$bmi <-  factor(risk_estimate$bmi)
risk_estimate$brstproc <-  factor(risk_estimate$brstproc)
risk_estimate$density <-  factor(risk_estimate$density)
risk_estimate$lastmamm <-  factor(risk_estimate$lastmamm)
risk_estimate$hrt <-  factor(risk_estimate$hrt)
risk_estimate$Hispanic <-  factor(risk_estimate$Hispanic)
risk_estimate$agefirst <-  factor(risk_estimate$agefirst)
risk_estimate$surgmeno<-  factor(risk_estimate$surgmeno)
risk_estimate$cancer<- as.numeric(risk_estimate$cancer)

risk_estimate_premenpaus <- risk_estimate %>% filter(menopaus!= "9" & menopaus!= "1")
risk_estimate_postmenpaus <- risk_estimate %>% filter(menopaus != "9" & menopaus != "0")



# train_pre = subset(risk_estimate_premenpaus, training == "1") # train pre data
# train_post = subset(risk_estimate_postmenpaus, training == "1") # train post data


### CAret Package work
data_set_size <- floor(nrow(risk_estimate_premenpaus)/1.5)
indexes <- sample(1:nrow(risk_estimate_premenpaus), size = data_set_size)
# Assign the data to the correct sets

training_predata <- risk_estimate_premenpaus[indexes,]
testing_predata <- risk_estimate_premenpaus[-indexes,]

training_predata$agegrp <- factor(training_predata$agegrp)
training_predata$brstproc <-  factor(training_predata$brstproc)
training_predata$nrelbc <- factor(training_predata$nrelbc)
training_predata$density <- factor(training_predata$density)
training_predata$cancer = as.factor(training_predata$cancer)
training_predata$count = as.numeric(training_predata$count)
table(training_predata$cancer )

data_balanced_under <- ovun.sample(cancer ~ ., data = training_predata, method = "under", N = 1814, seed = 1)$data
table(data_balanced_under$cancer)


set.seed(3333)
controlroc <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3,
                           savePredictions = "final",
                           classProbs = TRUE, index = createResample(data_balanced_under$cancer, 5),
                           selectionFunction = "oneSE", (levels(data_balanced_under$cancer)=c("No","Yes")),
                           summaryFunction = twoClassSummary, allowParallel = TRUE,verboseIter = TRUE)


model_listworks <- caretList(
  cancer ~ agegrp + brstproc + nrelbc + density,
  data = data_balanced_under,
  trControl=controlroc,
  metric="ROC",
  tuneList=list(
    gbmmodel = caretModelSpec(method = "gbm",train.fraction = 0.5,distribution="adaboost"),
    knnmodel = caretModelSpec(method = "knn"),
    glmmodel = caretModelSpec(method="glm", family = "binomial"),
    enetmodel=caretModelSpec(method="glmnet"),
    # rfmodel = caretModelSpec(method="rf", tuneGrid = expand.grid (.mtry=c(1:3)), ntrees = 100)
    rfmodel = caretModelSpec(method="rf")
  )
)

res <- resamples(model_listworks)
summary(res)
bwplot(res)


#modelCor(resamples(model_listworks))
greedy_ensemble <- caretEnsemble( model_listworks,  metric="ROC", trControl=controlroc)
summary(greedy_ensemble)

varImp(greedy_ensemble)



###############################################################
########################### POSTMENOPAUSE
##########################
##################################################################



### CAret Package work
data_set_size <- floor(nrow(risk_estimate_postmenpaus)/1.5)
indexes <- sample(1:nrow(risk_estimate_postmenpaus), size = data_set_size)
# Assign the data to the correct sets

training_postdata <- risk_estimate_postmenpaus[indexes,]
testing_postdata <- risk_estimate_postmenpaus[-indexes,]

training_postdata$agegrp <- factor(training_postdata$agegrp)
training_postdata$brstproc <-  factor(training_postdata$brstproc)
training_postdata$nrelbc <- factor(training_postdata$nrelbc)
training_postdata$density <- factor(training_postdata$density)
training_postdata$cancer = as.factor(training_postdata$cancer)
training_postdata$count = as.numeric(training_postdata$count)
table(training_postdata$cancer )


data_balanced_under <- ovun.sample(cancer ~ ., data = training_postdata, method = "under", N = 9900, seed = 1)$data
table(data_balanced_under$cancer)



set.seed(333334)
controlroc <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3,
                           savePredictions = "final",
                           classProbs = TRUE, index = createResample(data_balanced_under$cancer, 5),
                           selectionFunction = "oneSE", (levels(data_balanced_under$cancer)=c("No","Yes")),
                           summaryFunction = twoClassSummary, allowParallel = TRUE,verboseIter = TRUE)


model_listworks <- caretList(
  cancer~ agegrp + Hispanic + race +
    bmi + agefirst + brstproc + nrelbc +
    hrt + surgmeno + lastmamm + density,
  data = data_balanced_under,
  trControl=controlroc,
  metric="ROC",
  tuneList=list(
    gbmmodel = caretModelSpec(method = "gbm",train.fraction = 0.5,distribution="adaboost"),
    knnmodel = caretModelSpec(method = "knn"),
    glmmodel = caretModelSpec(method="glm", family = "binomial"),
    enetmodel=caretModelSpec(method="glmnet"),
    # rfmodel = caretModelSpec(method="rf", tuneGrid = expand.grid (.mtry=c(1:3)), ntrees = 100)
    rfmodel = caretModelSpec(method="rf")
  )
)

res <- resamples(model_listworks)
summary(res)
bwplot(res)


#modelCor(resamples(model_listworks))
greedy_ensemble <- caretEnsemble( model_listworks,  metric="ROC", trControl=controlroc)
summary(greedy_ensemble)

varImp(greedy_ensemble)



