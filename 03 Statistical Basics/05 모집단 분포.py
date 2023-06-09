# -*- coding: utf-8 -*-
"""
□ 참고 : 파이썬으로 배우는 통계학 교과서
□ 목적 : 모집단 분포
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
data = pd.read_csv('3-4-1-fish_length_100000.csv')

data_fish = data['length']
print(len(data_fish))
#%%
"""표본"""
# 표본추출(비복원)
sampling_result = np.random.choice(data_fish, size=10, replace=False)
sampling_result

# 표본평균
np.mean(sampling_result)
#%%
"""모집단"""
# 모평균
np.mean(data_fish)
# 모표준편차
np.std(data_fish)
# 모분산
np.var(data_fish)
# 모집단 히스토그램
sns.distplot(data_fish,kde=False, color='black')
plt.show()
#%%