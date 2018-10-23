#!/usr/bin/python3
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import pickle

clf=svm.SVC(gamma="scale")
iris=datasets.load_iris()
X, y=iris.data, iris.target
clf.fit(X, y)

#saving and loading model
s=pickle.dumps(clf)
clf2=pickle.loads(s)
print(clf2.predict(X[0:1]))

print(y[0])

#more efficient with big data
joblib.dump(clf, "filename.joblib")
clf=joblib.load("filename.joblib")

