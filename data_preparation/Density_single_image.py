import numpy as np
import os
import matplotlib.image as mpimg
import scipy.io as sio
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import math

def gaussian_filter_density(gt):
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    density = np.zeros(gt.shape, dtype=np.float32)
    type(distances)
    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=gt.shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        # starttime = datetime.datetime.now()
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
        # endtime = datetime.datetime.now()
        #
        # interval = (endtime - starttime)
        # print(interval)
    return density


def create_density(gts, d_map_h, d_map_w):
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if (bool_res[k] == True):
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    map_shape = [d_map_h, d_map_w]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        # starttime = datetime.datetime.now()
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
    return density

if __name__ == '__main__':
    Image_ = 'IMG_1.jpg'
    GT_Image = 'GT_IMG_1.mat'

    output_path = r'Single_result/'

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    img = mpimg.imread(Image_)
    gt  = sio.loadmat(GT_Image)
    gts = gt['image_info'][0][0][0][0][0]
    print('gts shape = {}'.format(gts.shape))
    
    if len(img.shape) < 3:
        img = img.reshape([img.shape[0], img.shape[1], 1])
        print('image.shape = {}'.format(img.shape))
    else:
        print('image shape = {}'.format(img.shape))
    
    density_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
    density_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

    density_map = create_density(gts / 4, density_map_h, density_map_w)
    print('density_map shape is {}'.format(density_map.shape))    
    
    ph = math.floor(float(img.shape[0] / 3.0))
    pw = math.floor(float(img.shape[1] / 3.0))
    
    d_map_ph = math.floor(math.floor(ph / 2.0) / 2.0)
    d_map_pw = math.floor(math.floor(pw / 2.0) / 2.0)
    count = 1 
    py1 = 1
    py2 = 1
    for i in range(1, 4):
        px1 = 1
        px2 = 1
        for j in range(1, 4):
            print('px = {}, py = {}'.format(px1, py1))
            final_image = img[py1 - 1: py1 + ph - 1, px1 - 1: px1 + pw - 1, :]
            final_gt = density_map[py2 - 1: py2 + d_map_ph - 1, px2 - 1: px2 + d_map_pw - 1]
            print('img shape is {}\ngt shape is {}'.format(final_image.shape, final_gt.shape))
            px1 = px1 + pw
            px2 = px2 + d_map_pw
            
            if final_image.shape[2] < 3:
                final_image = np.tile(final_image, [1, 1, 3])
            #image_final_name = output_path + '{}_{}.jpg'.format(i, j)
            #gt_final_name = output_path + '{}_{}.npy'.format(i, j)
            
            image_final_name = output_path + '{}.jpg'.format(count)
            gt_final_name = output_path + '{}.npy'.format(count)
            
            Image.fromarray(final_image).convert('RGB').save(image_final_name)
            np.save(gt_final_name, final_gt)
            count = count + 1
            #plt.imshow(final_gt)
            #plt.show()
            
        py1 = py1 + ph
        py2 = py2 + d_map_ph
    


plt.imshow(density_map)
plt.show()







