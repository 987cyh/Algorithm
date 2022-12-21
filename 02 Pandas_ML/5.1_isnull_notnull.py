# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# deck 열의 NaN 개수 계산하기
nan_deck = df['deck'].value_counts(dropna=False) 
print(nan_deck)

# isnull() 메서드로 누락 데이터 찾기
print(df.head().isnull())

# notnull() 메서드로 누락 데이터 찾기
print(df.head().notnull())

# isnull() 메서드로 누락 데이터 개수 구하기
print(df.isnull().sum(axis=0))
print(df.notnull().sum(axis=0))
#%%
