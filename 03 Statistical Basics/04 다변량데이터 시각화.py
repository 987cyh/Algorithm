# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 다변량 데이터 시각화
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
# 데이터 로드
data = pd.read_csv('3-3-2-fish_multi_2.csv')

length_a = data.query('species == "A"')["length"]
length_b = data.query('species == "B"')["length"]
#%%
# 히스토그램
sns.histplot(length_a, bins=5, color='black', kde=False)
sns.histplot(length_b, bins=5, color='gray', kde=False)
plt.show()

# 박스플롯
sns.boxplot(x='species',y='length',data=data,color='yellow')
plt.show()

# 바이올린플롯
sns.violinplot(x='species',y='length',data=data,color='green')
plt.show()

# 막대그래프
sns.barplot(x='species',y='length',data=data,color='green')
plt.show()

# 산포도
data1 = pd.read_csv('3-2-3-cov.csv')
sns.jointplot(x='x',y='y',data=data1,color='green')
plt.show()

# 페어플롯
data2 = sns.load_dataset('iris')
sns.pairplot(data=data2, hue='species')
plt.show()
#%%