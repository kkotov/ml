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
and rarely consider categorical predictors.

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
require(MASS)

covar <- t( matrix(c(1.00,0.90,0.70,
                     0.90,1.00,0.90,
                     0.70,0.90,1.00),ncol=3) )


df <- data.frame( mvrnorm( 10000, c(0,0,0), covar ) )
colnames(df) <- c("V1","V2","V3")

df$V3 <- factor( ifelse( df$V3>=0, rep(1,length(df$V3)), rep(-1,length(df$V3)) ) )

write.csv(file="one.csv",x=df)

require(ggplot2)
ggplot(df, aes(x=V1, y=V2, type=V3, color=V3)) + geom_point(size=0.1)
```
<img class=center src=one.png>

The file can be read processed with [example.cc](https://github.com/kkotov/ml/blob/master/example.cc) code:
```
g++ -Wl,--no-as-needed -g -Wall -std=c++11 -o rf example.cc -lpthread
./rf
```
resulting in the following printout:

Classification performance: 
    | -1  |  1	
----|-----|-----
-1: | 2092| 395	
1:  | 363 | 2152	

Regression performance: 
bias = 0.0125006 sd = 0.406995

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

    | -1  |  1
----|-----|----
 -1 | 2175| 310
  1 | 308 | 2207

[1] "bias =  -0.01926 , sd =  0.5055"

## Bibliography:

["The elements of Statistical Learning"](http://web.stanford.edu/~hastie/ElemStatLearn/)
