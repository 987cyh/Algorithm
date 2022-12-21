# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
import pandas as pd

# read_json() 함수로 데이터프레임 변환 
df = pd.read_json('./read_json_sample.json')
print(df)
print(df.index)
#%%
