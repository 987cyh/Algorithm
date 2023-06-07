# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 피어슨 상관계수(Pearson Correlation Coefficient) 복습
"""
#%%
# package
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
#%%
# 데이터 로드
cov_data = pd.read_csv('3-2-3-cov.csv')

# 데이터 분리
x = cov_data['x']
y = cov_data['y']

# 표본크기
N = len(cov_data)

# 평균값 계산
mu_x = np.mean(x)
mu_y = np.mean(y)

# 공분산 계산
cov_sample = sum((x - mu_x) * (y - mu_y)) / N
cov = sum((x - mu_x) * (y - mu_y)) / (N - 1)

# 분산계산
sigma_2_x = np.var(x, ddof=1)
sigma_2_y = np.var(y, ddof=1)

# 상관계수
rho = cov / np.sqrt(sigma_2_x * sigma_2_y)

np.corrcoef(x,y)
#%%