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
        self.stringList = []
        self.stringList2 = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.loadData(trainFile,testFile)
        
    # Runs the standard set of tests
    def run(self):
        self.extractTrainData()
        self.extractTestData()
        X_tra, X_te, y_tra, y_te = cross_validation.train_test_split(
            self.x_train, self.y_train, test_size=0.3, random_state=0)
        self.fitModel(X_tra, y_tra)
        print(self.scoreModel(X_te, y_te))


    # Loads data from files to lists
    def loadData(self, trainFile, testFile):
        f = open(trainFile, 'rb')
        try:
            reader = csv.reader(f)
            for row in reader:
                self.stringList.append(row)
        finally:
            f.close()
                
        f2 = open(testFile, 'rb')
        try:
            reader = csv.reader(f2)
            for row in reader:
                self.stringList2.append(row)
        finally:
            f2.close()

    # Exctracts usefull features from the raw train data.
    def extractTrainData(self):
        for s in self.stringList[1:]:
#            x_train.append([s[2],s[4],s[5],s[6],s[7],s[9],s[11],s[10],s[3]])
            self.x_train.append([s[2]])
            self.y_train.append(int(s[1]))

    # Exctracts usefull features from the raw test data.
    def extractTestData(self):
        for s in self.stringList2[1:]:
#            x_test.append([s[1],s[3],s[4],s[5],s[6],s[8],s[10],s[9],s[2]])
            self.x_test.append([s[1]])
            self.y_test.append(int(s[0]))

    # Fit the model to the training data
    def fitModel(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    # Score the model
    def scoreModel(self, x_test, y_test):
        return self.clf.score(x_test, y_test)
        
