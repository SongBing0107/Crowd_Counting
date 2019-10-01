import numpy as np
import cv2
import os
import random
import pandas as pd

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        # print('Now path is {}'.format(os.getcwd()))
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample # bool
        self.pre_load = pre_load           # bool
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        # print('data_files[0] = {}'.format(self.data_files[0]))
        # 是否打亂
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)

        self.num_samples = len(self.data_files) # 樣本數
        print('num of samples = {}'.format(self.num_samples))


        self.blob_list = {}
        self.id_list = range(0,self.num_samples)
        if self.pre_load: # 查看有無事先處理過?
            print('Pre-loading the data. This may take a while...')
            idx = 0
            ## self.data_files = os.listdir(data_path)
            for fname in self.data_files:
                img = cv2.imread(os.path.join(self.data_path, fname), 0)
                # print('reading path = {}\nImage shape = {}'.format(os.path.join(self.data_path, fname), img.shape))
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht/4)*4
                wd_1 = (wd/4)*4

                img = cv2.resize(img, (int(wd_1), int(ht_1)))
                img = img.reshape((1, 1, img.shape[0], img.shape[1]))
                # print('csv file path is {}'.format(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv')))
                # data = np.load()
                # print('path {}'.format(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv')))
                # print(os.path.isfile(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy')))

                data = np.load(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy'))
                den = pd.DataFrame(data)
                den = den.values

                # den = np.load(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy'))
                # den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None, error_bad_lines=False).as_matrix()

                den = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1/4
                    ht_1 = ht_1/4
                    #den = cv2.resize(den, (int(wd_1), int(ht_1)))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                    
                den = den.reshape((1, 1, den.shape[0], den.shape[1]))
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 100 == 0:                    
                    print('Loaded ', idx, '/', self.num_samples, 'files')
               
            print('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:  
                pass          
                # random.shuffle(self.id_list)        
            else:
                pass
                # random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht/4)*4
                wd_1 = (wd/4)*4
                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                #den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                #den  = den.astype(np.float32, copy=False)
                den = np.load(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy'))
                den = pd.DataFrame(data)
                den - den.values
                if self.gt_downsample:
                    wd_1 = wd_1/4
                    ht_1 = ht_1/4
                    den = cv2.resize(den,(wd_1,ht_1))                
                    den = den * ((wd*ht)/(wd_1*ht_1))
                else:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                    
                den = den.reshape((1,1,den.shape[0],den.shape[1]))            
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            
        
