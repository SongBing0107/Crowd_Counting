import scipy.io as sio
import cv2
import numpy as np
import os 
import sys

if __name__ == '__main__':
    test_path = '../data/original/shanghaitech/true_crowd_counting/train_gt'
    print(os.path.isdir(test_path))