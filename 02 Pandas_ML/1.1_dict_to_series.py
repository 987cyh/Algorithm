# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
import pandas as pd

# k:v 구조를 갖는 딕셔너리를 만들고, 변수 dict_data에 저장

dict_data = {'az': 100, 'bx': 2000, 'cv': 30000}

# 판다스 Series() 함수로 딕셔너리(dict_data)를 시리즈로 변환. 변수 sr에 저장 
sr = pd.Series(dict_data)

# 변수 sr의 자료형 출력
print(type(sr))

# 변수 sr에 저장되어 있는 시리즈 객체를 출력
print(sr)
#%%
