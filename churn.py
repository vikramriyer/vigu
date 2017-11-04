@author: Gaurav

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("c:\\Users\\gauravh\\vigu\\train.csv")
#data.columns.get_loc("HNW_CATEGORY")
data=data.dropna(axis=1,how='all')
data=data.dropna(axis=0,how='all')
data.drop('city',axis=1,inplace=True)
data.drop('zip',axis=1,inplace=True)

frac = len(data) * 0.8
data=data.dropna(thresh=frac, axis=1)
data.shape

null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()

obj=data.select_dtypes(include=[object]).columns
obj

lis=[]
for x in obj:
    lis.append(data.columns.get_loc(x))
lis    

for col in data.columns.values:
    if data[col].dtypes != 'object':
        print(col)
        data[col].fillna(data[col].mean(), inplace=True)
        
for col in data.columns.values:
    if data[col].dtypes == 'object':
        print(col)
        data[col].fillna(data[col].value_counts().index[0], inplace=True)
        
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()

labelencoder = LabelEncoder()

for col in data.columns.values:
    if data[col].dtypes == 'object':
        data[col] = labelencoder.fit_transform(data[col])

onehotencoder = OneHotEncoder(categorical_features=lis)
data = onehotencoder.fit_transform(data).toarray()

sc_X = StandardScaler()
data = sc_X.fit_transform(data)
