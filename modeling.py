#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("DataCleaned.csv")

features = [
 'Term',
 'NoEmp',
 'NewExist',
 'UrbanRural',
 'RevLineCr',
 'LowDoc',
 'GrAppv',
 'ZipCity',
 'ZipState',
 'NAICS_new',
 'ISCreateJob',
 'ISFranchise'
]

target = 'MIS_Status'

X_train = df[features]
y_train = df[target]

#Build preprocessor for the features
numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])
numeric_features = [
                    'Term', 
                    'NoEmp', 
                    'GrAppv', 
    ]

categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
categorical_features =  [
                     'NewExist',
                     'UrbanRural',
                     'RevLineCr',
                     'LowDoc',
                     'ZipCity',
                     'ZipState',
                     'NAICS_new',
                     'ISCreateJob',
                     'ISFranchise'
    ]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    steps=[("preprocessor", preprocessor), ("xgb", XGBClassifier(gamma = 0, 
                                                                 learning_rate = 0.1, 
                                                                 n_estimators = 300,
                                                                 max_depth = 5,
                                                                 reg_lambda = 1, 
                                                                 min_child_weight = 3,
                                                                 subsample = 0.9,
                                                                 random_state = 42))]).fit(X_train, y_train)

model = model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

