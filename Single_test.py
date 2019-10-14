import os 
import torch
import numpy as np
import h5py

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False 

output = True
outpath = r'./test_analysis'
analysis = True

if not os.path.isdir(outpath):
    os.mkdir(outpath)

model_path = r'result_1001/mcnn_shtechA_348.h5' # result_1001 best model
model_name = os.path.basename(model_path).split('.')[0]

img_name = r''
gt_name = r''

analysis_report = os.path.join(outpath, 'Analysis_' + img_name + '.txt')
print('output analysis report path is {}'.format(analysis_report))

net = CrowdCounter()
trained_model = os.path.join(modek_path)

# read model's parameter 
h5f = h5.py.File(trained_model, mode='r')

for k, v in net.state_dict().items():
    param = torch.from_numpy(np.asarray(h5f[k]))
    v.copy_(param)
net.cuda()
net.eval()
































