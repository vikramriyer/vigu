"""
@author: GauravH
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("c:\\Users\\gauravh\\vigu\\train.csv")

train_x = data.iloc[:, :-1]

train_y = data.iloc[:,-1].values

train_x=train_x.dropna(axis=1,how='all')
train_x=train_x.dropna(axis=0,how='all')
train_x.drop('city',axis=1,inplace=True)
train_x.drop('zip',axis=1,inplace=True)

frac = len(train_x) * 0.7
train_x=train_x.dropna(thresh=frac, axis=1)
train_x.shape

null_columns=train_x.columns[train_x.isnull().any()]
train_x[null_columns].isnull().sum()

obj=train_x.select_dtypes(include=[object]).columns
obj

lis=[]
for x in obj:
    lis.append(train_x.columns.get_loc(x))
lis   

for col in train_x.columns.values:
    if train_x[col].dtypes != 'object':
        print(col)
        train_x[col].fillna(train_x[col].mean(), inplace=True)
        
for col in train_x.columns.values:
    if train_x[col].dtypes == 'object':
        print(col)
        train_x[col].fillna(train_x[col].value_counts().index[0], inplace=True)
        
null_columns=train_x.columns[train_x.isnull().any()]
train_x[null_columns].isnull().sum()

labelencoder = LabelEncoder()

for col in train_x.columns.values:
    if train_x[col].dtypes == 'object':
        train_x[col] = labelencoder.fit_transform(train_x[col])
        
onehotencoder = OneHotEncoder(categorical_features=lis)
train_x = onehotencoder.fit_transform(train_x).toarray()

sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)
