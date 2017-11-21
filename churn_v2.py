# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:37:13 2017

@author: GauravH
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, metrics, ensemble, neighbors, linear_model, tree, model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from numpy import array,array_equal
from sklearn import cross_validation as cv
import xgboost as xgb
data = pd.read_csv("c:\\Users\\gauravh\\vigu\\train.csv")

data['Responders'].value_counts()

train_x = data.iloc[:, :-1]
train_y = data.iloc[:,-1].values
train_d = data.iloc[:,-1]

train_x=train_x.dropna(axis=1,how='all')
train_x=train_x.dropna(axis=0,how='all')
train_x.drop('city',axis=1,inplace=True)
train_x.drop('zip',axis=1,inplace=True)

frac = len(train_x) * 0.7
train_x=train_x.dropna(thresh=frac, axis=1)
train_x.shape


for c in train_x.columns:
    if train_x[c].dtype == 'object' and c not in ["Responders", "UCIC_ID"]:
        lbl = LabelEncoder()
        lbl.fit(list(train_x[c].values.astype('str')))
        train_x[c] = lbl.transform(list(train_x[c].values.astype('str')))
       
for col in train_x.columns.values:
    if train_x[col].dtypes != 'object' and train_x[col].isnull().sum() > 0:
        train_x[col].fillna(train_x[col].mean(), inplace=True)
        
# Remove constant features

def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

constant_features_train = set(identify_constant_features(train_x))
train_x.drop(constant_features_train, inplace=True, axis=1)
len(constant_features_train)

# Remove equals features
def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

equal_features_train = identify_equal_features(train_x)

features_to_drop = array(equal_features_train)[:,1] 
train_x.drop(features_to_drop, axis=1, inplace=True)


skf = cv.KFold(n=300000,n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}

def score_model(model):
    return cv.cross_val_score(model, train_x, train_d, cv=skf, scoring=score_metric)

scores['tree'] = score_model(tree.DecisionTreeClassifier()) 

scores['ada_boost'] = score_model(ensemble.AdaBoostClassifier())

#{'ada_boost': array([ 0.8437071 ,  0.84604536,  0.84229304]),
 #'tree': array([ 0.67651593,  0.6743412 ,  0.68035429])}
