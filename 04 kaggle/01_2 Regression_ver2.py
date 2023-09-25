# -*- coding: utf-8 -*-
"""
목적: kaggle 데이터를 통한 알고리즘 공부
데이터: Used Cars Price Prediction / https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction
주제: 회귀분석의 이해
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
#%%
# 데이터 불러오기
data = pd.read_csv('train-data.csv')

# 데이터 구성확인
data.head()
data.info()
data.shape
data.describe()
#%%
# 데이터 전처리
data.duplicated().sum() # 중복값 확인
data.isnull().sum() # 누락값 확인
data.info()

# value 내 단위 제거
cols_error = ['Mileage', 'Engine', 'Power']
for col in cols_error:
    # data[col] = data[col].apply(lambda x: float(str(x).split()[0]))
    data[col] = data[col].apply(lambda x: str(x).split()[0])
data['Power'] = data['Power'].apply(lambda x: None if x == 'null' else x) # flaot 형식변경이 깨져서 확인

# 변수형식변경
cols_error = ['Mileage', 'Engine', 'Power']
for col in cols_error:
    data[col] = data[col].astype(float)
data.info()

# 불필요 칼럼제거
data.isnull().sum() # 누락값 확인
data.drop(columns=['Unnamed: 0','New_Price'],inplace=True)

# 분류 기준 생성 : 1876 → 31
len(data['Name'].unique().tolist())
data['Company'] = data['Name'].apply(lambda x: x.split()[0]) # 기준
len(data['Name'].unique().tolist())
len(data['Company'].unique().tolist())
data.drop(columns='Name', inplace=True)

# 변수간의 상관관계 검토
plt.figure(figsize=(10,10))
sns.heatmap(data=data[['Price','Kilometers_Driven','Engine','Power','Mileage']].corr(), annot=True)
plt.show()

# 결측값 처리: 평균 값으로 대체
data[['Price','Kilometers_Driven','Engine','Power','Mileage']].describe()
data.info()
data.isnull().sum() # 누락값 확인
data['Engine'] = data['Engine'].fillna(data['Power'].mean())
data['Power'] = data['Power'].fillna(data['Power'].mean())

data['Mileage'] = data['Mileage'].apply(lambda x: np.nan if x == 0 else x)  # 이상치 : Mileage(연비) 값 0
data['Mileage'] = data['Mileage'].fillna(data['Mileage'].mean())
data['Seats'] = data['Seats'].apply(lambda x: np.nan if x == 0 else x) # 이상치 : Seats의 값 0
data['Seats'] = data['Seats'].fillna(data['Seats'].mean())
data['Seats'] = data['Seats'].apply(lambda x: int(round(x,0))) # 시트의 수는 정수
data.isnull().sum() # 누락값 확인

# 이상치 전처리
data.info()
data.describe()
data[['Price','Kilometers_Driven','Engine','Power','Mileage']].describe()
# 시각화
fig, ax = plt.subplots(1,5, figsize=(16,4))
ax[0].boxplot([data['Kilometers_Driven']])
ax[1].boxplot([data['Engine']])
ax[2].boxplot([data['Power']])
ax[3].boxplot([data['Mileage']])
ax[4].boxplot([data['Price']])
ax[0].set_title('Kilometers_Driven')
ax[1].set_title('Engine')
ax[2].set_title('Power')
ax[3].set_title('Mileage')
ax[4].set_title('Price')
plt.show()
# pairplot
sns.pairplot(data=data,x_vars=['Kilometers_Driven','Engine','Power','Mileage'],y_vars=['Price'],size=3)
plt.show()

# 제거
data = data[data['Kilometers_Driven'] < data['Kilometers_Driven'].max()] # 6500000
# pairplot
sns.pairplot(data=data,x_vars=['Kilometers_Driven','Engine','Power','Mileage'],y_vars=['Price'],size=3)
plt.show()
#%%
# 정규화
data['Kilometers_Driven'].hist() # skewness(왜도)를 제거, Log Transform
plt.show()
data['Engine'].hist() # skewness(왜도)를 제거, Log Transform
plt.show()
data['Power'].hist() # skewness(왜도)를 제거, Log Transform
plt.show()
data['Price'].hist() # skewness(왜도)를 제거, Log Transform
plt.show()
data['Mileage'].hist() # 정규화되어 있음
plt.show()

for col in ['Kilometers_Driven','Engine','Power','Price']:
    data['Log_' + col] = np.log(data[col])
    data.drop(columns=col, inplace=True)

data.info()
# 범주형변수의 더미변수화
for col in ['Location','Fuel_Type', 'Transmission', 'Owner_Type', 'Year', 'Seats', 'Company']:
    temp = pd.get_dummies(data[col], prefix=col, drop_first=True)
    data.drop(columns=col, inplace=True)
    data = pd.concat([data, temp], axis=1)
#%%
# 독립, 종속변수 설정
data.info()
x = data.drop(columns='Log_Price')
y = data['Log_Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10) # 8:2

import statsmodels.api as sm
x = sm.add_constant(x) # 상수항 추가
model = sm.OLS(y, x)
result = model.fit()
result.summary()
#%%
# 회귀분석 종류별 비교
from sklearn.metrics import mean_squared_error # MSE

# Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression() # LinearRegression(fit_intercept=True)가 기본값이며, 상수함 포함
model.fit(x_train, y_train)
pred = model.predict(x_train)
print(model.intercept_) # 추정된 상수항
print(model.coef_) # 추장된 가중치 백터

model_pred_train = model.predict(x_train)
model_pred_test = model.predict(x_test)
print('LinearRegression train R-squared: ',model.score(x_train, y_train))
print('LinearRegression test R-squared : ',model.score(x_test, y_test))
print('LinearRegression train MSE      : ',mean_squared_error(y_train, model_pred_train))
print('LinearRegression test MSE       : ',mean_squared_error(y_test, model_pred_test))

plt.scatter(np.exp(y_test), np.exp(model_pred_test))
plt.plot([-1, 100], [-1, 100], 'r')
plt.xlabel('y')
plt.ylabel('pred_y')
plt.title('Linear Regression')
plt.show()
#%%
# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 10)
ridge.fit(x_train, y_train)
print(ridge.intercept_)
print(ridge.coef_)

ridge_pred_train = ridge.predict(x_train)
ridge_pred_test = ridge.predict(x_test)

print('Ridge train R-squared: ',ridge.score(x_train, y_train))
print('Ridge test R-squared : ',ridge.score(x_test, y_test))
print('Ridge train MSE      : ',mean_squared_error(y_train, ridge_pred_train))
print('Ridge test MSE       : ',mean_squared_error(y_test, ridge_pred_test))

plt.scatter(np.exp(y_test), np.exp(ridge_pred_test))
plt.plot([-1, 100], [-1, 100], 'r')
plt.xlabel('y')
plt.ylabel('pred_y')
plt.title('Ridge')
plt.show()
#%%
# Lasso Regression
from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.001)
lasso.fit(x_train, y_train)
print(lasso.intercept_)
print(lasso.coef_)

lasso_pred_train = lasso.predict(x_train)
lasso_pred_test = lasso.predict(x_test)

print('lasso train R-squared: ',lasso.score(x_train, y_train))
print('lasso test R-squared : ',lasso.score(x_test, y_test))
print('lasso train MSE      : ',mean_squared_error(y_train, lasso_pred_train))
print('lasso test MSE       : ',mean_squared_error(y_test, lasso_pred_test))

plt.scatter(np.exp(y_test), np.exp(lasso_pred_test))
plt.plot([-1, 100], [-1, 100], 'r')
plt.xlabel('y')
plt.ylabel('pred_y')
plt.title('Lasso')
plt.show()
#%%
# 모델평가
print('LinearRegression test R-squared : ',model.score(x_test, y_test))
print('LinearRegression train MSE      : ',mean_squared_error(y_train, model_pred_train))

print('ridge test R-squared : ',ridge.score(x_test, y_test))
print('ridge train MSE      : ',mean_squared_error(y_train, ridge_pred_train))

print('lasso test R-squared : ',lasso.score(x_test, y_test))
print('lasso train MSE      : ',mean_squared_error(y_train, lasso_pred_train))
#%%
# 회귀분석 가정검토
# 1. 독립변수와 종속변수간의 선형성
sns.pairplot(data[['Log_Price','Log_Kilometers_Driven','Log_Engine','Log_Power','Mileage']])
plt.show()

# 2. 잔차의 정규성
import scipy.stats

residual = y_test - model_pred_test
sr = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-5, 5], [-5, 5], '--', color='red')
plt.show()

# 3. 잔차의 등분산성
sns.regplot(model_pred_test, np.sqrt(np.abs(sr)), lowess=True, line_kws={'color': 'red'})
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.show()

# 4. 다중공선성 : https://mindscale.kr/course/basic-stat-python/13/
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

model = ols('Log_Price ~ Log_Kilometers_Driven + Mileage + Log_Engine + Log_Power',
            data[['Log_Kilometers_Driven','Log_Engine','Log_Power','Log_Price','Mileage']])

res = model.fit()

pd.DataFrame({'feature': column, 'VIF': variance_inflation_factor(model.exog, i)}
             for i, column in enumerate(model.exog_names)
             if column != 'Intercept')
#%%
