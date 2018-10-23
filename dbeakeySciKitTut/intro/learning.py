#!/usr/bin/python3
# full code http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
from sklearn import svm
from sklearn import datasets

digits=datasets.load_digits()
#SVM classifier
clf=svm.SVC(gamma=.001, C=100.)
#learning based on last column
clf.fit(digits.data[:-1], digits.target[:-1])

#prediction
print(clf.predict(digits.data[-1:]))
