# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 기본 통계 복습
"""
#%%
# package
import numpy as np
import scipy as sp
from scipy import stats
#%%
# 1변량 데이터
data = np.array([2,3,3,4,4,4,4,5,5,6])

# 평균값(기대값)
N = len(data)
sum_value = np.sum(data)
mu = sum_value / N
print(mu)
print(np.mean(data))

# 표본분산
sigma_2_sample = np.sum((data - mu) ** 2) / N
print(sigma_2_sample)
print(np.var(data, ddof=0))

# 불편분산
sigma_2 = np.sum((data - mu) ** 2) / (N - 1)
print(sigma_2)
print(np.var(data, ddof=1))

# 표준편차
sigma = np.sqrt(sigma_2)
print(sigma)
print(np.std(data, ddof=1))

# 표준화
np.mean(data - mu)
np.std(data / sigma, ddof=1)

standard = (data - mu) / sigma
np.mean(standard)
np.std(standard, ddof=1)

# 사분위수
data1 = np.array([1,2,3,4,5,6,7,8,9])
stats.scoreatpercentile(data1,25)
stats.scoreatpercentile(data1,75)
#%%