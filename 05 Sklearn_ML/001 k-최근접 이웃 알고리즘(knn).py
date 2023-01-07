# -*- coding: utf-8 -*-
"""
목적: 다시 한번 머신러닝 개념 정리
참고도서: 파이썬 라이브러리를 활용한 머신러닝
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
#%%
# 데이터셋 로드
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#%%
# 훈련 및 검증데이터 분류
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names) # columns=iris_dataset['feature_names']
pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o', hist_kwds={'bins':20}, s=60)
plt.show()
#%%
# 모델선정 및 훈련
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#%%
# 예측하기
X_new = np.array([[5,2.9,1,0.2]])
X_new.shape

prediction = knn.predict(X_new)
prediction
iris_dataset['target_names'][0]
#%%
# 평가하기
y_pred = knn.predict(X_test)
y_pred

np.mean(y_pred==y_test)
knn.score(X_test,y_test)
#%%
# 훈련 및 평가
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#%%
