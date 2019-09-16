import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

output_dir = './result/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)