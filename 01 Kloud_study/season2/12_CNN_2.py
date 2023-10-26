# -*- coding: utf-8 -*-
"""
@author(Director of Research) : Kloud80

@student: 987cyh

내용 : Convolution Network 이해 강의용
"""
# package
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
#%%
#구글 드라이브에 연결
from google.colab import drive
drive.mount('/content/gdrive/')
#작업 폴더 이동
os.chdir('/content/gdrive/My Drive/Colab Notebooks/urban_data_mining_23/07 CNN/data/')
os.listdir()
# os.getcwd()
#%%
#이미지를 불러내서 numpy로 변환
img = Image.open('images/BV1171010400100090001.png')
data = np.array(img)

plt.figure(figsize=(500/80, 500/80))
plt.imshow(data)

#numpy 구조
data.shape

#채널별 데이터 보기
data[:,:, 3]

"""색깔별 출력"""
plt.figure(figsize=(500/80, 500/80))
plt.imshow(data[:, : , 0], cmap='Reds_r')


plt.figure(figsize=(500/80, 500/80))
plt.imshow(data[:, : , 1], cmap='Greens_r')


plt.figure(figsize=(500/80, 500/80))
plt.imshow(data[:, : , 2], cmap='Blues_r')

#빨간색 채널 변경
data[0:200, :, 0] = 255
plt.figure(figsize=(500/80, 500/80))
plt.imshow(data)

#초록색 채널 변경
data[:, 0:200, 1] = 255
plt.figure(figsize=(500/80, 500/80))
plt.imshow(data)

#파란색 채널 변경
data[-200:-1, :, 2] = 255
plt.figure(figsize=(500/80, 500/80))
plt.imshow(data)

#회색 처리
data_gray = np.zeros([500,500])
gray = 0.2989 * data[:,:,0] + 0.5870 * data[:,:,1] + 0.1140 * data[:,:,2]
data_gray = gray
data_gray = data_gray.astype('int')

plt.figure(figsize=(500/80, 500/80))
plt.imshow(data_gray, cmap='gray')

#이미지 생성하기
r = np.array([[255, 255, 255, 255, 255],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [255, 192, 128, 64, 0],
              [0, 64, 128, 192, 255]])
g = np.array([[0, 0, 0, 0, 0],
              [255, 255, 255, 255, 255],
              [0, 0, 0, 0, 0],
              [0, 64, 128, 192, 255],
              [0, 0, 0, 0, 0]])
b = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [255, 255, 255, 255, 255],
              [0, 0, 0, 0, 0],
              [255, 192, 128, 64, 0]])
t = np.array([[255, 192, 128, 64, 0],
              [255, 192, 128, 64, 0],
              [255, 192, 128, 64, 0],
              [255, 255, 255, 255, 255],
              [255, 255, 255, 255, 255]])

img = np.zeros([5,5,4]).astype('int')
img[:,:,0] = r
img[:,:,1] = g
img[:,:,2] = b
img[:,:,3] = t

plt.imshow(img)
#%%