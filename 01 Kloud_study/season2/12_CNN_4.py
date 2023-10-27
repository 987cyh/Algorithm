# -*- coding: utf-8 -*-
"""
@author(Director of Research) : Kloud80

@student: 987cyh

내용 : 로드뷰 건물 벽면 이미지 구별하기 CNN 학습
"""

# package
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import keras.layers as layers
import keras.optimizers as optimizers
from keras.models import Model, load_model
from keras import layers
from keras import models
from keras.models import load_model
import ipywidgets
#%%
#구글 드라이브에 연결
from google.colab import drive
drive.mount('/content/gdrive/')
#작업 폴더 이동
os.chdir('/content/gdrive/My Drive/Colab Notebooks/urban_data_mining_23/07 CNN/data/')
os.listdir()
# os.getcwd()
#%%
#벽돌과 대리석 이미지 라벨링 불러오기
data = pd.read_csv('data.csv', sep=',', encoding='cp949')
#y값을 지정
data['y'] = data['type'].apply(lambda x: 1 if x =='대리석' else 0)
data.head(10)

#벽돌과 대리석 이미지  불러오기
flist = 'images/' + data['image']

image_list = []
for f in flist:
    img = Image.open(f)
    image_list.append(np.array(img))

#학습데이터 100개
training_x = np.array(image_list[:100])
training_y = np.array(data['y'].values[:100])

#검증 데이터 나머지
testing_x = np.array(image_list[100:])
testing_y = np.array(data['y'].values[100:])

print('training x : ' + str(training_x.shape) + ', y : ' + str(training_y.shape))
print('testing x : ' + str(testing_x.shape) + ', y : ' + str(testing_y.shape))

#출력해보기
print(training_y[0])
plt.imshow(training_x[0])
plt.show()

#컬러 값을 0~1로 노멀라이징
training_x = training_x / 255
testing_x = testing_x / 255
#%%
#CNN 네트워크 만들기
inputs = layers.Input(shape=(500, 500, 4))
net = layers.Conv2D(10, kernel_size=5, padding='same')(inputs)
net = layers.LeakyReLU()(net)
net = layers.MaxPool2D(pool_size=5)(net)
net = layers.Conv2D(10, kernel_size=5, padding='same')(net)
net = layers.LeakyReLU()(net)
net = layers.MaxPool2D(pool_size=5)(net)
net = layers.Conv2D(20, kernel_size=3, padding='same')(net)
net = layers.LeakyReLU()(net)
net = layers.MaxPool2D(pool_size=5)(net)
net = layers.Conv2D(25, kernel_size=2, padding='same')(net)
net = layers.LeakyReLU()(net)
net = layers.MaxPool2D(pool_size=4)(net)
net = layers.Flatten()(net)
net = layers.Dense(10, activation='relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(1, activation='sigmoid')(net)

model = Model(inputs=inputs, outputs=net)
model.compile(
    loss='binary_crossentropy',  #mse, mean_absolute_error
    optimizer='adam',
    metrics=['acc']
)
model.summary()

hist = model.fit(training_x, training_y, epochs=100)

model = load_model('save-0.1606-0.1034.h5')
model.summary()
#%%
#학습에 사용된 데이터를 이용하여 예측한다
predict_y = model.predict(training_x, verbose=1)

#실제값과 예측값 shape를 동일하게 변경
training_y = training_y.reshape(100,1)
predict_y.shape

result_training = np.concatenate([training_y, predict_y], axis=1)

result_training = np.round(result_training, 3)
# result_training[np.where(result_training<0.5)] = 0
# result_training[np.where(result_training>=0.5)] = 1
print(result_training)
#%%
#학습 내용 출력용 함수
def diplay_result(idx = 0) :
  global training_x, training_y, predict_y
  tx = training_x.copy()
  tx = tx * 255
  tx = tx.astype('int')

  dpi = 80
  img = training_x[idx, :, :, :]
  y = training_y[idx, 0]
  yhat = predict_y[idx, 0]

  figsize = 500/float(dpi), 500/float(dpi)
  fig = plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.title('Real : '+ {y==0:'bricks',y==1:'marble'}.get(True) + ' // Predict : ' + {yhat<0.5:'bricks',yhat>0.5:'marble'}.get(True))

#위젯을 이용하여 학습 결과 출력하기
ipywidgets.interact(diplay_result, idx=(0, training_y.shape[0], 1) )

#검증 데이터를 이용하여 예측한다
predict_y2 = model.predict(testing_x, verbose=1)

#실제값과 예측값 shape를 동일하게 변경
testing_y = testing_y.reshape(38,1)
predict_y2.shape
result_testing = np.concatenate([testing_y, predict_y2], axis=1)

result_testing = np.round(result_testing, 3)
# result_testing[np.where(result_testing<0.5)] = 0
# result_testing[np.where(result_testing>=0.5)] = 1
print(result_testing)
#%%
#검증 내용 출력용 함수
def diplay_result_test(idx = 0) :
  global testing_x, testing_y, predict_y2
  tx = testing_x.copy()
  tx = tx * 255
  tx = tx.astype('int')

  dpi = 80
  img = testing_x[idx, :, :, :]
  y = testing_y[idx, 0]
  yhat = predict_y2[idx, 0]

  figsize = 500/float(dpi), 500/float(dpi)
  fig = plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.title('Real : '+ {y==0:'bricks',y==1:'marble'}.get(True) + ' // Predict : ' + {yhat<0.5:'bricks',yhat>0.5:'marble'}.get(True))


#위젯을 이용하여  검증 결과 출력하기
ipywidgets.interact(diplay_result_test, idx=(0, testing_y.shape[0], 1) )
#%%
#학습 내용 출력용 함수
def diplay_result_layer(layer=1, idx = 0, f_map=0) :
  global model, training_x, training_y, predict_y

  new_model = Model(model.input,model.layers[layer].output) #각 층별로 레이어를 자른다
  predict = new_model.predict(np.array([training_x[idx, :, :, :]])) #dataseq위치의 학습 데이터를 입력한다

  dpi = 80
  if f_map > predict.shape[3] : f_map =  predict.shape[3]
  img = predict[0, :, :, f_map]

  figsize = 500/float(dpi), 500/float(dpi)
  fig = plt.figure(figsize=figsize)
  plt.imshow(img, cmap='gray')
  plt.title(model.get_layer(index=layer))
  plt.show()

#위젯을 이용하여  검증 결과 출력하기
ipywidgets.interact(diplay_result_layer, layer=(1, len(model.layers)-6, 1), f_map=(0,20,1), idx=(0, testing_y.shape[0], 1) )
#%%
#flatten layer 확인
new_model = Model(model.input,model.layers[15].output) #flatten 결과 비교
predict = new_model.predict(np.array([training_x[dataseq]]))
print(predict)

predict = new_model.predict(training_x)
result_flatten = np.concatenate((result_training, predict), axis=1)

result_flatten = pd.DataFrame(result_flatten)

result_flatten

np_flatten = np.array(result_flatten.values)
np_flatten.shape

#flatten 값 클러스터링
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10).fit(np_flatten[:, 1:])
result_kmean = np.array(kmeans.labels_)
result_kmean

#클러스터링 별로 출력해 보기
view_cluster = 1
f_cls = np.where(result_kmean == view_cluster)[0]
fig = plt.figure(figsize=(15,10))

for idx in range(len(f_cls)) :
  img = training_x[f_cls[idx],:,:,:] * 255
  img = img.astype('int')
  ax = fig.add_subplot(int((len(f_cls)) / 5+1), 5, idx+1)
  ax.imshow(img)
  ax.axis('off')

plt.show()
#%%