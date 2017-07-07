# Random Forest

## Introduction

This package is yet another Random Forest implementation. Why implementing again?
A quick review of several off-the-shelf packages uncovered they all have the same
[fatal flaw](http://www.drdobbs.com/windows/a-brief-history-of-windows-programming-r/225701475).
Apart from jokes, I have two rather specific requirements in our [EMTF](https://kkotov.github.io/emtf.html)
system. They are:
1) real-time application: prediction should be quick and made on event-by-event basis
2) light-weight standalone C++ code: easy to integrate in already existing project
The first one renders useless batch-processing system and the second disfavors
all-in-one machine learning libraries. Why still not use available light-weight C++
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
4) [RandomForest.h](https://github.com/kkotov/ml/blob/master/RandomForest.h)

## Example snippets

Convenience _DataFrame_ class is made to 


## Bibliography:

["The elements of Statistical Learning"](http://web.stanford.edu/~hastie/ElemStatLearn/)
