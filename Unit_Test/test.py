import scipy.io as sio
import cv2
import numpy as np
import os 
print(os.getcwd())
print(os.listdir())

os.chdir('./Unit_Test')
from .data_preparation.Get_Density_Map_Gaussian import Get_Density_Map_Gaussian

dataname = 'GT_IMG_5.mat'
data = sio.loadmat(dataname)
datainfo = data['image_info'][0, 0][0, 0][0]

print(datainfo.shape)

Imagename = 'IMG_5.jpg'
Image = cv2.imread(Imagename)

cv2.imshow('origin image', Image)
# cv2.imwrite('origin image.jpg', Image)

'''
for x, y in datainfo:
    # print('x = {}, y = {}'.format(x, y))
    cv2.circle(Image, (int(x), int(y)), 10, (0, 0, 255), 0)

# cv2.imshow('img', Image)
# cv2.imwrite('labeled image.jpg', Image)
'''
Image2 = Get_Density_Map_Gaussian(Image, datainfo)
cv2.imshow('Image2', Image2)



cv2.waitKey(0)
cv2.destroyAllWindows()