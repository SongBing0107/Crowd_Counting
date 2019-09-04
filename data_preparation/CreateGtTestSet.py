import numpy as np
import cv2
import os 
import scipy.io as sio

Dataset = 'A'
Dataset_Name = 'shanghaitech_part_' + Dataset
path = '../data/original/shanghaitech/part_' + Dataset + '/test_data/images/'
gt_path = '../data/original/shanghaitech/part_' + Dataset + '/test_data/ground-truth/'
gt_path_csv = '../data/original/shanghaitech/part_' + Dataset + '/test_data/ground_truth_csv/'

if not os.path.isdir(gt_path_csv):
    os.mkdir(gt_path_csv)

num_images = 182 if Dataset == 'A' else 316


for i in range(1, num_images + 1):
    if i % 10 == 0:
        pass
        # print('Pricessing {}/{} files'.format(i, num_images))
    
    gt_name = gt_path + "GT_IMG_" + str(i) + '.mat'
    gt_Info = sio.loadmat(gt_name)
    
    if i == 1:
        print(type(gt_Info['image_info']))
        data = gt_Info.get('image_info')
        print(data)

    input_image_name = path + "IMG_" + str(i) + '.jpg'
    Image = cv2.imread(input_image_name)
    (H, W, C) = Image.shape

    if C == 3:
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)

    annPoints = 0
    # Image_Density = get_density_map_gaussian(Image, annPoints)


