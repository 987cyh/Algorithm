# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
# 라이브러리 불러오기
import pandas as pd

# 중복 데이터를 갖는 데이터프레임 만들기
df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                  'c2':[1, 1, 1, 2, 2],
                  'c3':[1, 1, 2, 2, 2]})
print(df)

# 데이터프레임 전체 행 데이터 중에서 중복값 찾기
df_dup = df.duplicated()
print(df_dup)
print(df.duplicated().sum()) # 중복행 수

# 데이터프레임의 특정 열 데이터에서 중복값 찾기
col_dup = df['c2'].duplicated()
print(col_dup)
#%%
