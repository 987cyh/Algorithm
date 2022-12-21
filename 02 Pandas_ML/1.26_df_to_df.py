# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
# 라이브러리 불러오기
import pandas as pd
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
print(df.tail())          #마지막 5행을 표시
print(type(df))

# 데이터프레임에 숫자 10 더하기
addition = df + 10
print(addition.tail())    #마지막 5행을 표시
print(type(addition))

# 데이터프레임끼리 연산하기 (additon - df)
subtraction = addition - df
print(subtraction.tail())   #마지막 5행을 표시
print(type(subtraction))
#%%
