# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 모집단분포와 정규분포 간 확률밀도함수 비교
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
# 데이터
data = pd.read_csv('3-4-1-fish_length_100000.csv')
data_fish = data['length']

x = np.arange(start=1, stop=7.1, step=0.1)
print(x)

# 확률밀도
stats.norm.pdf(x=x, loc=4, scale=0.8)
plt.plot(x, stats.norm.pdf(x=x, loc=4, scale=0.8), color='black')
plt.show()

sns.distplot(data_fish,kde=False, norm_hist=True, color='black')
plt.plot(x, stats.norm.pdf(x=x, loc=4, scale=0.8), color='black')
plt.show()
#%%
