# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 표본평균, 모평균 복습
"""
#%%
# package
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#%%
# 표본평균
population = stats.norm(loc=4, scale=0.8)

sample_mean_array = np.zeros(10000)

np.random.seed(1)
for k in range(0, 10000):
    sample = population.rvs(size=10)
    sample_mean_array[k] = np.mean(sample)

print(sample_mean_array)
print(np.mean(sample_mean_array))
print(np.std(sample_mean_array))
sns.distplot(sample_mean_array,color='black')
plt.show()
#%%
size_array = np.arange(start=10, stop=100100, step=100)
print(size_array)

sample_mean_array_size = np.zeros(len(size_array))
print(sample_mean_array_size)

np.random.seed(1)
for k in range(0, len(size_array)):
    sample = population.rvs(size=size_array[k])
    sample_mean_array_size[k] = np.mean(sample)

plt.plot(size_array, sample_mean_array_size,color='black')
plt.xlabel('sample size')
plt.ylabel('sample mean')
plt.show()
#%%
# 표본평균 계산 사용자 정의 함수
def calc_sample_mean(size, n_trial):
    sample_mean_array = np.zeros(n_trial)
    for i in range(0, n_trial):
        sample = population.rvs(size = size)
        sample_mean_array[i] = np.mean(sample)
    return(sample_mean_array)

np.random.seed(1)
np.mean(calc_sample_mean(size = 10, n_trial = 10000))
#%%

np.random.seed(1)
# samle size 10
size_10 = calc_sample_mean(size = 10, n_trial = 10000)
size_10_df = pd.DataFrame({
    "sample_mean":size_10,
    "size"       :np.tile("size 10", 10000)
})
# samle size 20
size_20 = calc_sample_mean(size = 20, n_trial = 10000)
size_20_df = pd.DataFrame({
    "sample_mean":size_20,
    "size"       :np.tile("size 20", 10000)
})
# samle size 30
size_30 = calc_sample_mean(size = 30, n_trial = 10000)
size_30_df = pd.DataFrame({
    "sample_mean":size_30,
    "size"       :np.tile("size 30", 10000)
})
# 종합
sim_result = pd.concat(
    [size_10_df, size_20_df, size_30_df])
# 결과
print(sim_result.head())

sns.violinplot(x = "size", y = "sample_mean", data = sim_result, color = 'gray')
plt.show()
#%%