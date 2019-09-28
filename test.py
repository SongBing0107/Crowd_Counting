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
vis = False 
output = True

data_path = 'data/original/shanghaitech/part_A/test_data/images'
gt_path = 'data/original/shanghaitech/part_A/test_data/ground-truth'
model_path = 'result/mcnn_shtechA_2000.h5'
model_name = os.path.basename(model_path).split('.')[0]

outpath = './output/'  
result_report = os.path.join(outpath, 'density_map_report_' + model_name)

if not os.path.isdir(outpath):
    os.mkdir(outpath)
if not os.path.isdir(result_report):
    os.mkdir(result_report)

net = CrowdCounter()
trained_model = os.path.join(model_path)

h5f = h5py.File(trained_model, mode='r')

for k, v in net.state_dict().items():
    param = torch.from_numpy(np.asarray(h5f[k]))
    v.copy_(param)

net.cuda()
net.eval()

mae, mse = 0, 0

dataloader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for blob in dataloader:
    im = blob['data']
    gt = blob['gt_density']

    density_map = net(im, gt)
    density_map = density_map.cpu().numpy()

    gt_count = np.sum(gt)
    et_count = np.sum(density_map)
    
    mae = mae + abs(gt_count - et_count)
    mse = mse + (gt_count - et_count) * (gt_count - et_count)

    if output:
        utils.save_density_map(density_map, outpath, 'output_' + blob['fname'].split('.')[0] + '.png')

mae = mae / dataloader.get_num_samples()
mse = np.sqrt(mse / dataloader.get_num_samples())
print('mae = {}, mse = {}'.format(mae, mse))







    





