# Random Forest

## Introduction

This package is yet another Random Forest implementation.
A quick review of several off-the-shelf packages uncovered that they all have the same
[fatal flaw](http://www.drdobbs.com/windows/a-brief-history-of-windows-programming-r/225701475).
Apart from jokes, I have two rather specific requirements in our [EMTF](https://kkotov.github.io/emtf.html)
system. They are:
1) real-time application: prediction should be quick and made on event-by-event basis
2) light-weight stand alone C++ code: easy to integrate in already existing project

The first one renders useless batch-processing system while the second disfavors
all-in-one machine learning libraries. Why still not to use available light-weight C++
packages, that generations of CS students implement and put on github? It turns out
that most of those are limited to either classification or regression (not both)
and rarely consider categorical predictors. On the other hand my version also has
limitations as it does not, for example, support *ordinal* categorical variables.

## Design

All of the code is inlined in four header files:
1) [DataFrame.h](https://github.com/kkotov/ml/blob/master/DataFrame.h) - a two
dimensional table representing columns of continuous and categorical variables
2) [Tree.h](https://github.com/kkotov/ml/blob/master/Tree.h) - binary tree
3) [TreeTrainer.h](https://github.com/kkotov/ml/blob/master/TreeTrainer.h) - code
that finds a series of best splits and builds a decision/regression tree
4) [RandomForest.h](https://github.com/kkotov/ml/blob/master/RandomForest.h) -
main user front-end and host of few handles 

## Example snippets

A test sample with three correlated variables (two continuous and one categorical)
can be produced with the following R script:
```
phi <- runif(10000,0,2*3.1415927)
rho <- rnorm(10000,phi*0.3,phi*0.05)
x1 <- rho*cos(phi)
y1 <- rho*sin(phi)
x2 <- rho*cos(phi-3.1415927*2/3)
y2 <- rho*sin(phi-3.1415927*2/3)
x3 <- rho*cos(phi-3.1415927*4/3)
y3 <- rho*sin(phi-3.1415927*4/3)

df <- data.frame(V1 = c(x1,x2,x3),
                 V2 = c(y1,y2,y3),
                 V3 = factor(c(rep(1,length(x1)),
                               rep(2,length(x2)),
                               rep(3,length(x3))
                             )
                      )
      )

write.csv(file="one.csv",x=df)

require(ggplot2)
ggplot(df,aes(x=V1,y=V2,type=V3,color=V3)) + geom_point(shape = 1, size = 0.1)

```
<img class=center src=one.png>

The file can be read processed with [example.cc](https://github.com/kkotov/ml/blob/master/example.cc) code:
```
g++ -Wl,--no-as-needed -g -Wall -std=c++11 -o rf example.cc -lpthread
./rf
```
resulting in the following printout:

Classification performance: 

&nbsp; |  1  |  2  |  3
-------|-----|-----|------
1      | 4748| 162 | 103
2      | 105 | 4738| 157
3      | 149 | 100 | 4740

Regression performance: 

bias = -0.00122876 sd = 0.657455

It can be compared to the [ranger's](https://github.com/imbs-hl/ranger) results with
```
require(ranger)

trainSet <- df[seq(1,nrow(df),2),]
testSet  <- df[seq(2,nrow(df),2),]

modelFit1 <- ranger("V3 ~ V1 + V2", data=trainSet)

table(testSet$V3,  predict(modelFit1,testSet)$prediction)

modelFit2 <- ranger("V1 ~ V2 + V3", data=trainSet)

m <- mean(testSet$V1 - predict(modelFit2,testSet)$prediction)

sdev <- sd(testSet$V1 - predict(modelFit2,testSet)$prediction)

print(paste("bias = ", signif(m,4), ", sd = ", signif(sdev,4)))
```
resulting in the following printout:

&nbsp; |  1  |  2  |  3
-------|-----|-----|------
1      | 4746| 114 | 140
2      | 148 | 4732| 120
3      | 117 | 142 | 4741

[1] "bias =  0.0006507 , sd =  0.6773"

## Bibliography:

["The elements of Statistical Learning"](http://web.stanford.edu/~hastie/ElemStatLearn/)
