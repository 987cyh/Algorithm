# -*- coding: utf-8 -*-
"""
@author(Director of Research) : Kloud80

@student: 987cyh
"""

# package
import pandas as pd
import numpy as np
import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, concatenate, Add, Conv1D, Flatten
from keras import optimizers
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
#%%
# 구글 드라이브에 연결
from google.colab import drive
drive.mount('/content/gdrive/')
# 작업 폴더 이동
os.chdir('/content/gdrive/My Drive/Colab Notebooks/urban-data-mining/06 CNN/data/')
os.listdir()
# os.getcwd()
#%%
data = pd.read_csv('생활인구_학습데이터.txt', sep='|', encoding='cp949')
data.dtypes
data['TOT_REG_CD'] = data['TOT_REG_CD'].astype('str')
data = data.fillna(0.0)

data['18년6월'].hist(bins=100, figsize=(20, 10))
data['21년6월'].hist(bins=100, figsize=(20, 10))

print('18년6월 100명 미만 집계구 수 : ' + str(data[data['18년6월'] < 100].shape[0]))
print('21년6월 100명 미만 집계구 수 : ' + str(data[data['21년6월'] < 100].shape[0]))

data = data[data['18년6월'] >= 100]
data = data[data['21년6월'] >= 100]

data['18년6월'].hist(bins=100, figsize=(20, 10))
data['21년6월'].hist(bins=100, figsize=(20, 10))

# 데이터 분포
data['인구차이'].hist(bins=100, figsize=(20, 10))

# 이상치 제거
data = data[(data['인구차이'] < -0.01) | (data['인구차이'] > 0.01)]
data['인구차이'].hist(bins=100, figsize=(20, 10))
data['인구차이'] = data['인구차이'].apply(lambda x: 0.02 if x > 0.02 else x)  # 최대값 지정
data['인구차이'] = data['인구차이'].apply(lambda x: -0.02 if x < -0.02 else x)  # 최소값 지정

# 학습 데이터셋 만들기

tmp = data.dtypes
tmp = data.mean()
learning = data[['아파트', '단독주택', '다가구주택', '다세대주택', '다중주택', '연립주택',
                 '오피스텔', '사무소', '기타일반업무시설',
                 '상점', '가설점포', '소매점', '일반음식점', '기타제1종근린생활시설', '기타제2종근린생활시설', '기타판매시설',
                 '학원', '고시원', '독서실', '대학교', '기타교육연구시설',
                 '공',
                 '인구차이']].copy()

learning['주택'] = learning['아파트'] + learning['단독주택'] + learning['다가구주택'] + learning['다세대주택'] + learning['연립주택']
learning['업무'] = learning['오피스텔'] + learning['사무소'] + learning['기타일반업무시설']
learning['상가'] = learning['상점'] + learning['가설점포'] + learning['소매점'] + learning['일반음식점'] + learning['기타제1종근린생활시설'] + \
                 tmp['기타제2종근린생활시설']
learning['교육'] = learning['학원'] + learning['고시원'] + learning['독서실'] + learning['대학교'] + learning['기타교육연구시설']
learning['공원'] = learning['공']
learning = learning[['주택', '업무', '상가', '교육', '공원', '인구차이']]
learning.head(10)

# 학습 데이터 numpy에 할당
x_train = np.array(learning[['주택', '업무', '상가', '교육', '공원']].values)
y_train = np.array(learning[['인구차이']].values)

def result_display(y, y_hat):
    print("R square : " + str(r2_score(y, y_hat)))

    plt.scatter(y, y_hat)
    plt.title('predict result')
    plt.ylabel('y_hat')
    plt.xlabel('y_true')
    plt.show()


# 이전 시간 Dense 분석 결과 불러오기
model = load_model('dense.h5')
model.summary()

ypred = model.predict(x_train)
result_display(y_train, ypred)

inputs = Input(shape=(5, 1))
net = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(inputs)
net = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(net)
net = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(net)
net = Flatten()(net)
net = Dense(units=1, activation='linear')(net)

c_model = Model(inputs, net)
c_model.summary()

#%%
c_model.compile(loss='mse', optimizer='adam')
history = c_model.fit(x_train, y_train, epochs=3000)

ypred2 = c_model.predict(x_train)
result_display(y_train, ypred2)
#%%