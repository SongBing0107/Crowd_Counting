import os 
import cv2
import pandas as pd
import numpy as np

train_img_path = r'train_img/IMG_15_1.jpg'
train_gt_path = r'train_gt/IMG_15_1.npy'

img = cv2.imread(train_img_path, 0)
print('img.shape = {}'.format(img.shape))
print(img)

img = img.astype(np.float32, copy=False)
h, w = img.shape
h1 = (h * 4) / 4
w1 = (w * 4) / 4

img = cv2.resize(img, (int(w1), int(h1)))
print('img.shape = {}'.format(img.shape))

img = img.reshape((1, 1, img.shape[0], img.shape[1]))
print('img.shape = {}'.format(img.shape))


data = np.load(train_gt_path)
print('npload data.shape = {}'.format(data.shape))
print(data)

data = pd.DataFrame(data)
print(data)
print(type(data))
print('///////////////////////////////')

data.astype(np.float32, copy=False)
ndata = data.values
print(ndata)
print(ndata.shape)
print(type(ndata))

downsample = True
# downsample = False 

if downsample:
    print('h1 = {}, w1 = {}'.format(h1, w1))
    h1 = h1 / 4
    w1 = w1 / 4
    print('h1 = {}, w1 = {}'.format(h1, w1))
    ndata = cv2.resize(ndata, (int(w1), int(h1)))
    ndata = ndata * ((w * h) / (w1 * h1)) 
    print(ndata)
    print(ndata.shape)
else:
    ndata = cv2.resize(ndata, (int(w1), int(h1)))
    ndata = ndata * ((w * h) / (w1 * h1))
    print(ndata)
    print(ndata.shape)

ndata = ndata.reshape((1, 1, ndata.shape[0], ndata.shape[1]))
print('Input image shape is {}'.format(img.shape))
print('Input gt shape is {}'.format(ndata.shape))

blob = {}
blob['data'] = img
blob['gt_density'] = ndata
blob['fname'] = 'IMG_15_1.jpg'






