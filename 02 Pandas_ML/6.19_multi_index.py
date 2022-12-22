# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%

# 라이브러리 불러오기
import pandas as pd
import seaborn as sns

# titanic 데이터셋에서 age, sex 등 5개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','sex', 'class', 'fare', 'survived']]

# class 열, sex 열을 기준으로 분할
grouped = df.groupby(['class', 'sex'])  

# 그룹 객체에 연산 메서드 적용
gdf = grouped.mean()
print(gdf)
print('\n')
print(type(gdf))

# class 값이 First인 행을 선택하여 출력
print(gdf.loc['First'])
print('\n')

# class 값이 First이고, sex 값이 female인 행을 선택하여 출력
print(gdf.loc[('First', 'female')])
print('\n')

# sex 값이 male인 행을 선택하여 출력
print(gdf.xs('male', level='sex'))
#%%
"""
ㅁ 참고: https://wikidocs.net/158257
"""
data = {'col1':[0,1,2,3,4], 'col2':[5,6,7,8,9],
        'level0':['A','A','A','B','B'],
        'level1':['X','X','Y','Y','Z'],
        'level2':['a','a','b','c','a']}
df0 = pd.DataFrame(data=data)
df0 = df0.set_index(['level0', 'level1', 'level2'])
df0.index

print(df0)
print(df0.xs(key='A'))
print(df0.xs(key=('A','X')))
print(df0.xs(key='Y',level='level1')) # level을 지정하여 하위분류
print(df0.xs(key='Y',level='level1',drop_level=False)) # drop_level=True로 할 경우 key값으로 지정된 레벨을 포함
#%%
