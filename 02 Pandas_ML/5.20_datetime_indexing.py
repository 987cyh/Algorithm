# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 파일 읽어와서 df로 변환
df = pd.read_csv('stock-data.csv')

# 문자열인 날짜 데이터를 판다스 Timestamp로 변환
df['new_Date'] = pd.to_datetime(df['Date'])   # 새로운 열에 추가
df.set_index('new_Date', inplace=True)        # 행 인덱스로 지정

print(df.head())
print(df.index)

# 날짜 인덱스를 이용하여 데이터 선택하기
df_y = df.loc['2018']
print(df_y.head())

df_ym = df.loc['2018-07']    # loc 인덱서 활용
print(df_ym)

df_ym_cols = df.loc['2018-07', 'Start':'High']    # 열 범위 슬라이싱
print(df_ym_cols)

df_ymd = df.loc['2018-07-02']
print(df_ymd)

# FutureWarning: Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys
#                is deprecated and will raise a KeyError in a future Version.
df_ymd_range = df.loc['2018-06-10':'2018-06-20']    # 날짜 범위 지정 → KeyError 발생 시킬 수 있음
print(df_ymd_range)

# 시간 간격 계산. 최근 180일 ~ 189일 사이의 값들만 선택하기
today = pd.to_datetime('2018-12-25')            # 기준일 생성
df['time_delta'] = today - df.index             # 날짜 차이 계산
df.set_index('time_delta', inplace=True)        # 행 인덱스로 지정
df_180 = df['180 days':'189 days']
print(df_180)
#%%
