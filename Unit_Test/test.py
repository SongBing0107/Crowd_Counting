import scipy.io as sio
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_path = '../data/original/shanghaitech/true_crowd_counting/train_gt/'
    ls = os.listdir(test_path)
    ls.sort()

    print(ls[0])
    temp = np.load(test_path + ls[0])
    plt.imshow(temp)
    plt.show()
    print(type(temp))