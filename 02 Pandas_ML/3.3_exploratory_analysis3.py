# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# 평균값 
print(df.mean())
print(df['mpg'].mean())
print(df.mpg.mean())
print(df[['mpg','weight']].mean())

# 중간값 
print(df.median())
print(df['mpg'].median())

# 최대값 
print(df.max())
print(df['mpg'].max())

# 최소값 
print(df.min())
print(df['mpg'].min())

# 표준편차 
print(df.std())
print(df['mpg'].std())

# 상관계수 
print(df.corr())
print(df[['mpg','weight']].corr()) # 피어슨 기본값
print(df[['mpg','weight']].corr(method='pearson'))  # 피어슨
print(df[['mpg','weight']].corr(method='kendall'))  # 켄달-타우
print(df[['mpg','weight']].corr(method='spearman')) # 스피어먼
#%%
