import scipy.io as sio
import cv2
import numpy as np
import os 
import sys
from  data_preparation.GetDensity import Get_Density_Map_Gaussian

try:
    print(os.getcwd())
    from data_preparation.GetDensity import Get_Density_Map_Gaussian
except ImportError:
    print('can not find the path')


dataname = 'GT_IMG_5.mat'
data = sio.loadmat(dataname)
datainfo = data['image_info'][0, 0][0, 0][0]

print(datainfo.shape)

Imagename = 'IMG_5.jpg'
Image = cv2.imread(Imagename)

cv2.imshow('origin image', Image)

print('Image shape = {}, datainfo shape = {}'.format(Image.shape, datainfo.shape))

# Image2 = Get_Density_Map_Gaussian(Image, datainfo)

# cv2.imshow('Image2', Image2)
cv2.waitKey(0)
cv2.destroyAllWindows()