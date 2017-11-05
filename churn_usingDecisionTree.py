# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/home/gaurav/Downloads/AV/churn-pred/vigu/train.csv")


data_test = pd.read_csv("/home/gaurav/Downloads/AV/churn-pred/vigu/test.csv")

train_x = data.iloc[:, :-1]
train_y = data.iloc[:,-1].values

#took all coumns except the label columns which is the last one
test_x = data_test.iloc[:, :-1]

#on train data, remove the city and zip columns
train_x=train_x.dropna(axis=1,how='all')
train_x=train_x.dropna(axis=0,how='all')
train_x.drop('city',axis=1,inplace=True)
train_x.drop('zip',axis=1,inplace=True)

#on test data, remove the same columns
test_x=test_x.dropna(axis=1,how='all')
test_x=test_x.dropna(axis=0,how='all')
test_x.drop('city',axis=1,inplace=True)
test_x.drop('zip',axis=1,inplace=True)

#Took those columns which have almost 70% of data 
frac = len(train_x) * 0.7
train_x=train_x.dropna(thresh=frac, axis=1)
train_x.shape

frac = len(test_x) * 0.7
test_x=test_x.dropna(thresh=frac, axis=1)
test_x.shape


null_columns=train_x.columns[train_x.isnull().any()]
train_x[null_columns].isnull().sum()

null_columns=test_x.columns[test_x.isnull().any()]
test_x[null_columns].isnull().sum()

obj=train_x.select_dtypes(include=[object]).columns
obj

lis=[]
for x in obj:
    lis.append(train_x.columns.get_loc(x))
lis


obj=test_x.select_dtypes(include=[object]).columns
obj

lis_test=[]
for x in obj:
    lis_test.append(test_x.columns.get_loc(x))
lis_test

for col in train_x.columns.values:
    if train_x[col].dtypes != 'object':
        train_x[col].fillna(train_x[col].mean(), inplace=True)
        
for col in test_x.columns.values:
    if test_x[col].dtypes != 'object':
        test_x[col].fillna(test_x[col].mean(), inplace=True)


for col in train_x.columns.values:
    if train_x[col].dtypes == 'object':
        train_x[col].fillna(train_x[col].value_counts().index[0], inplace=True)
        
        
null_columns=train_x.columns[train_x.isnull().any()]
train_x[null_columns].isnull().sum()

null_columns=test_x.columns[test_x.isnull().any()]
test_x[null_columns].isnull().sum()

labelencoder = LabelEncoder()

for col in train_x.columns.values:
    if train_x[col].dtypes == 'object':
        train_x[col] = labelencoder.fit_transform(train_x[col])
        
for col in test_x.columns.values:
    if test_x[col].dtypes == 'object':
        test_x[col] = labelencoder.fit_transform(test_x[col])
        
onehotencoder = OneHotEncoder(categorical_features=lis)
train_x = onehotencoder.fit_transform(train_x).toarray()

onehotencoder = OneHotEncoder(categorical_features=lis_test)
test_x = onehotencoder.fit_transform(test_x).toarray()

sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)

sc_X = StandardScaler()
test_x = sc_X.fit_transform(test_x)

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=5)
clf.fit(train_X, train_y)

test_pred = clf.predict(test_X)

from sklearn.metrics import accuracy_score
accuracy_score(test_y, test_pred)

#result come out to be accuracy_score(test_y, test_pred)
#Out[34]: 0.81605050505050503





