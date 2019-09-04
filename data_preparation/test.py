import scipy.io as sio

dataname = 'GT_IMG_1.mat'
data = sio.loadmat(dataname)


# 好像是要求data裡面data['image_info']的資料 不過那個資料結構有點看不懂== 
# 不太確定那個31行如果在python裡面要怎改寫 麻煩阿盧看一下ㄌ
print(data['image_info'])
print(type(data['image_info']))