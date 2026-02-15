Prediction Assignment Writeup - Module 8: Practical Machine Learning
================
Roddy Mendoza Marriott
2026-02-09

------------------------------------------------------------------------

**Background**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

------------------------------------------------------------------------

**The data**

-The training data for this project are available here:
\[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>\]

-The test data are available here:
\[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>\]

-The data for this project come from this source:
\[<http://groupware.les.inf.puc-rio.br/har>\]. If you use the document
you create for this class for any purpose please cite them as they have
been very generous in allowing their data to be used for this kind of
assignment.

------------------------------------------------------------------------

**Some insights about the project**

The goal of your project is to predict the manner in which they did the
exercise. This is the `classe` variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

------------------------------------------------------------------------

**Data processing**

First, loading the packages and libraries.

``` r
library(plyr)
library(dplyr)
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(kernlab)
library(randomForest)
library(knitr)
library(e1071)
```

------------------------------------------------------------------------

**Getting and cleaning data**

``` r
trainingst <- read.csv("pml-training.csv")
testingst <- read.csv("pml-testing.csv")
```

``` r
dim(trainingst)
```

    ## [1] 19622   160

``` r
dim(testingst)
```

    ## [1]  20 160

------------------------------------------------------------------------

**Preprocessing and cleaning data**

we should Exclude the obvious columns i.e `X`,
`user_name`,`raw_timestamp_part_1`,`raw_timestamp_part_2`,
`cvtd_timestamp`,`roll_belt` which are the first 7 columns. We should
also delete missing values and variables with near zero variance.

``` r
#Deleting missing values 
trainingst <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))  
testingst <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

``` r
#Deleting missing values
trainingst<-trainingst[,colSums(is.na(trainingst)) == 0]
testingst <-testingst[,colSums(is.na(testingst)) == 0]
```

``` r
#Removing columns that are not predictors, which are the the seven first columns
trainingst   <-trainingst[,-c(1:7)]
testingst <-testingst[,-c(1:7)]
```

``` r
dim(trainingst)
```

    ## [1] 19622    53

``` r
dim(testingst)
```

    ## [1] 20 53

From the above code block sum(completeCase) == nrows confirm that the
number of complete case is equal to number of rows in `trainingdf` same
for `testingdf`

Now we have only 53 columns(features) are left. we can preproccess the
training and testing i.e converting into scales of `0` to `1` and
replacing any `NA` values to average of that columns.

------------------------------------------------------------------------

**Partition the data set into `training` and `testing` data from
`trainingst`**

``` r
inTrain <- createDataPartition(y = trainingst$classe, p=0.75, list = FALSE)
training <- trainingst[inTrain, ]
testing <- trainingst[-inTrain, ]
```

------------------------------------------------------------------------

**Training the model**

Two methods will be applied to model, and the best one will be used for
the(testingst) predictions.

The methods are: `Decision Tree` and `Random Forests`.

**Model 1: Training the model with Decision Trees**

``` r
set.seed(40000)
fitDT <- rpart(classe ~ .,training, method="class")
# Normal plot
rpart.plot(fitDT)
```

![](Machine-learning-project_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
#Use model to predict classe in validation testing set
predictionDT <- predict(fitDT, testing, type = "class")
```

``` r
#Estimate the errors of the prediction algorithm in the Decision Tree model
cmdt <-confusionMatrix(as.factor(testing$classe), predictionDT)
cmdt
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1247   43   41   52   12
    ##          B  168  559  121   55   46
    ##          C   12   86  649   76   32
    ##          D   48   69   54  554   79
    ##          E   10   89   67   71  664
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.749           
    ##                  95% CI : (0.7366, 0.7611)
    ##     No Information Rate : 0.3028          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6819          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8397   0.6608   0.6964   0.6856   0.7971
    ## Specificity            0.9567   0.9039   0.9481   0.9390   0.9418
    ## Pos Pred Value         0.8939   0.5890   0.7591   0.6891   0.7370
    ## Neg Pred Value         0.9322   0.9274   0.9301   0.9380   0.9578
    ## Prevalence             0.3028   0.1725   0.1900   0.1648   0.1699
    ## Detection Rate         0.2543   0.1140   0.1323   0.1130   0.1354
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.8982   0.7823   0.8222   0.8123   0.8695

``` r
# Accuracy plot
plot(cmdt$table, col = cmdt$byClass, 
main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cmdt$overall['Accuracy'], 4)))
```

![](Machine-learning-project_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

**Model 2: Training the model using Random Forest**

``` r
rfModel <- randomForest(as.factor(classe)~., data=training)
# Summary of the model
rfModel
```

    ## 
    ## Call:
    ##  randomForest(formula = as.factor(classe) ~ ., data = training) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 7
    ## 
    ##         OOB estimate of  error rate: 0.42%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4182    3    0    0    0 0.0007168459
    ## B   13 2830    5    0    0 0.0063202247
    ## C    0   13 2552    2    0 0.0058433970
    ## D    0    0   20 2391    1 0.0087064677
    ## E    0    0    1    4 2701 0.0018477458

``` r
# Plot the variable importance
varImpPlot(rfModel)
```

![](Machine-learning-project_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
# Confusion matrix with testing
predTesting <- predict(rfModel, testing)
rfcfm  <- confusionMatrix(as.factor(testing$classe), predTesting)
rfcfm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    0    0    0    1
    ##          B    7  941    1    0    0
    ##          C    0    7  847    1    0
    ##          D    0    0    9  794    1
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9943          
    ##                  95% CI : (0.9918, 0.9962)
    ##     No Information Rate : 0.2857          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9928          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9950   0.9926   0.9883   0.9975   0.9978
    ## Specificity            0.9997   0.9980   0.9980   0.9976   0.9998
    ## Pos Pred Value         0.9993   0.9916   0.9906   0.9876   0.9989
    ## Neg Pred Value         0.9980   0.9982   0.9975   0.9995   0.9995
    ## Prevalence             0.2857   0.1933   0.1748   0.1623   0.1839
    ## Detection Rate         0.2843   0.1919   0.1727   0.1619   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9974   0.9953   0.9932   0.9975   0.9988

``` r
plot(rfcfm$table, col = rfcfm$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(rfcfm$overall['Accuracy'], 4)))
```

![](Machine-learning-project_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

**Remarks**

\-`Decision Tree Model` is the worst model running, it has the low mean
and the highest standard deviation.

\-`Random Forest Model` it has the highest mean accuracy and lowest
standard deviation.

Depending on how your model is to be used, the interpretation of the
kappa statistic might vary One common interpretation is shown as
follows:

• Poor agreement = Less than 0.20

• Fair agreement = 0.20 to 0.40

• Moderate agreement = 0.40 to 0.60

• Good agreement = 0.60 to 0.80

• Very good agreement = 0.80 to 1.00

This two models preforms as expected, the deviation from the cross
validation accuracy is low.

``` r
#plot the model
plot(rfModel)
```

![](Machine-learning-project_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

The predictive accuracy of the `Random Forest Model` is excellent at
99.8 %. Accuracy has plateaued, and further tuning would only yield
decimal gain.

**Making prediction on the 20 data pointsusing random forest**

`Decision Tree Model`: 73.43%, Random Forest Model: 99.53% The Random
Forest model is selected and applied to make predictions on the 20 data
points from the original testing dataset (testingst)

``` r
rfPredictions <- predict(rfModel, testingst,type= "class")
rfPredictions
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
