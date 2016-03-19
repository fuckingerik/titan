#The first forest
# Step by step, remember to check correlation between features

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
import re

from scipy.stats.stats import pearsonr

# @author eriset
# Class containing usefull functions and variables. 
# Ment to be used from an interactive python shell
class exploreData:

    # initializes the data object
    def __init__(self, trainFile, testFile):
        self.clf = ExtraTreesClassifier(n_estimators = 100, criterion = 'gini')
        f = open(trainFile, 'rb')
        self.stringList = []
        try:
            reader = csv.reader(f)
            for row in reader:
                stringList.append(row)
        finally:
            f.close()
                
        f2 = open(testFile, 'rb')
        self.stringList2 = []
        try:
            reader = csv.reader(f2)
            for row in reader:
                stringList2.append(row)
        finally:
            f2.close()
