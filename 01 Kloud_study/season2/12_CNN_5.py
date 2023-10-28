# -*- coding: utf-8 -*-
"""
@author(Director of Research) : Kloud80

@student: 987cyh

내용 : LENET
"""

# package
import pandas as pd
import numpy as np
import os, time, sys
import keras.layers as layers
import keras.optimizers as optimizers
from keras.models import Model, load_model
from keras import layers
from keras import models
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import ipywidgets
#%%
#구글 드라이브에 연결
from google.colab import drive
drive.mount('/content/gdrive/')
#작업 폴더 이동
os.chdir('/content/gdrive/My Drive/Colab Notebooks/urban_data_mining_23/07 CNN/')
os.listdir()
# os.getcwd()
#%%
# https://keras.io/api/datasets/mnist/
# 필기체 데이터 셋
(mnist_training_x, mnist_training_y), (mnist_testing_x, mnist_testing_y) = mnist.load_data()
assert mnist_training_x.shape == (60000, 28, 28)
assert mnist_testing_x.shape == (10000, 28, 28)
assert mnist_training_y.shape == (60000,)
assert mnist_testing_y.shape == (10000,)

idx = 4
plt.imshow(mnist_training_x[idx,:,:])
plt.title(mnist_training_y[idx])
plt.show()
#%%
# LENET 모델
# http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

inputs = layers.Input(shape=(32, 32, 1))
net = layers.Conv2D(6, kernel_size=5, strides=1, activation='tanh')(inputs)
net = layers.AveragePooling2D(pool_size=2, strides=2)(net)
# Combine Table이 있지만 구현하지 않음
net = layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh')(net)
net = layers.AveragePooling2D(pool_size=2, strides=2)(net)
#flatten 이 아니고 Conv 였음
net = layers.Conv2D(120, kernel_size=5, strides=1, activation='tanh')(net)
net = layers.Flatten()(net)
net = layers.Dense(84, activation='tanh')(net)
net = layers.Dense(10, activation='softmax')(net)

model = Model(inputs=inputs, outputs=net)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
model.summary()

# 28x28 mnist 이미지를 32x32 로 패딩 추가함
training_x = np.pad(mnist_training_x, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)
training_x.shape

#y 값을 sofrmax 학습하기 위해 더미변환
training_y = mnist_training_y.copy()
training_y = np_utils.to_categorical(training_y, 10)
training_y.shape

# y 값 확인
print(mnist_training_y[0])
print(training_y[0, :])

#모델 학습한다.
hist = model.fit(training_x, training_y, epochs=20, batch_size=128)

# 테스트 데이터로 모델 검증

loss,accuracy= model.evaluate(training_x, training_y)
print('training : loss = ' + str(loss) + ', accuracy = ' + str(accuracy))

loss,accuracy= model.evaluate(testing_x, testing_y)
print('testing : loss = ' + str(loss) + ', accuracy = ' + str(accuracy))

# 테스트 데이터 추론하기
testing_x = np.pad(mnist_testing_x, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)

testing_y = mnist_testing_y.copy()
testing_y = np_utils.to_categorical(testing_y, 10)

predict_y = model.predict(testing_x)

# 테스트 데이터 추론 결과 보기
idx = 1

plt.imshow(testing_x[idx, :, :])
plt.title('real = ' + str(mnist_testing_y[idx]) + ', predict = ' + str(np.where(predict_y[0,:] == predict_y[0,:].max())[0][0]))
plt.show()
#%%
#학습 내용 출력용 함수
def diplay_result_layer(layer=0, idx = 0, f_map=0) :
  global model, training_x, training_y, predict_y

  new_model = Model(model.input,model.layers[layer].output) #각 층별로 레이어를 자른다
  predict = new_model.predict(np.array([training_x[idx, :, :]])) #dataseq위치의 학습 데이터를 입력한다

  dpi = 80
  if f_map > predict.shape[3] : f_map =  predict.shape[3]
  img = predict[0, :, :, f_map]

  figsize = 500/float(dpi), 500/float(dpi)
  fig = plt.figure(figsize=figsize)
  plt.imshow(img, cmap='gray')
  plt.title(model.get_layer(index=layer))
  plt.show()

#위젯을 이용하여  검증 결과 출력하기
ipywidgets.interact(diplay_result_layer, layer=(0, 5, 1), f_map=(0,20,1), idx=(0, training_y.shape[0], 1) )
#%%