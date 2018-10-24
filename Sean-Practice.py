import numpy as np
from sklearn import svm

def main():
    f = open("filename.txt")
    f.readline()  # skip the header
    data = np.loadtxt(f)
    

if __name__ == '__main__':
    main()