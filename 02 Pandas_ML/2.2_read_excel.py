# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
import pandas as pd

# read_excel() 함수로 데이터프레임 변환 

df1 = pd.read_excel(fr'./남북한발전전력량.xlsx', engine='openpyxl')                 # header=0 (default 옵션)
df2 = pd.read_excel(fr'./남북한발전전력량.xlsx', engine='openpyxl', header=None)    # header=None 옵션

# 데이터프레임 출력
print(df1)
print(df2)
#%%
