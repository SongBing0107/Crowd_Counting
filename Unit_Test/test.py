import pandas as pd
import numpy as np

if __name__ == '__main__':
    # data = pd.read_csv('GT_IMG_15_8.npy')
    data =np.load('GT_IMG_15_8.npy')
    print(data.shape)
    for i in range(4):
        for j in range(4):
            print('data[{}][{}] = {}'.format(i, j, data[i, j]))

    data2 = pd.DataFrame(data)
    print(data2)
    print(data.shape)
