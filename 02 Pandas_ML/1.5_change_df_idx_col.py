# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
import pandas as pd

# 행 인덱스/열 이름 지정하여, 데이터프레임 만들기
df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']], 
                   index=['준서', '예은'],
                   columns=['나이', '성별', '학교'])

# 행 인덱스, 열 이름 확인하기
print(df)            #데이터프레임
print(df.index)      #행 인덱스
print(df.columns)    #열 이름

# 행 인덱스, 열 이름 변경하기
df.index=['학생1', '학생2'] # 재정의
df.columns=['연령', '남녀', '소속'] # 재정의

print(df)            #데이터프레임
print(df.index)      #행 인덱스
print(df.columns)    #열 이름
#%%
