# -*- coding: utf-8 -*-
"""
목적: kaggle 데이터를 통한 알고리즘 공부
데이터: Used Cars Price Prediction / https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
주제: 군집분석의 이해
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#%%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']
df.head()
#%%
X = df.iloc[:, :4]
Y = df['target']
x_train, x_valid, y_train, y_valid = train_test_split(df.iloc[:, :4], df['target'], stratify=df['target'], test_size=0.2, random_state=30)
x_train.shape, y_train.shape
x_valid.shape, y_valid.shape

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

prediction = knn.predict(x_valid)
(prediction == y_valid).mean()
knn.score(x_valid, y_valid)

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(x_train, y_train)
    score = knn.score(x_valid, y_valid)
    print('k: %d, accuracy: %.2f' % (k, score*100))
#%%