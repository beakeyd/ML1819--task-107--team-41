#!/usr/bin/python3
from sklearn import datasets

iris=datasets.load_iris()
digits=datasets.load_digits()

#features that can be accessed
print(digits.data)
#grount truth for dataset
print(digits.target)
