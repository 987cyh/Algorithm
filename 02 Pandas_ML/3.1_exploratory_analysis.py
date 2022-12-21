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

# 데이터프레임 df의 내용을 일부 확인 
print(df.head())     # 처음 5개의 행
print(df.tail())     # 마지막 5개의 행

# df의 모양과 크기 확인: (행(row)의 개수, 열(columns)의 개수)를 투플로 반환
print(df.shape)

# 데이터프레임 df의 내용 확인 
print(df.info())

# 데이터프레임 df의 자료형 확인 
print(df.dtypes)

# 시리즈(mog 열)의 자료형 확인 
print(df.mpg.dtypes)

# 데이터프레임 df의 기술통계 정보 확인 
print(df.describe())
print(df.describe(percentiles=[.01,.25,.75,.99])) # 분위수
print(df.describe(include='all'))
#%%
