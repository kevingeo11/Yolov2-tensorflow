'''
    This script displays the configuration parameters, training parameters and model architecture 
'''

import numpy as np
import tensorflow as tf
from os.path import join

import sys
sys.path.append(sys.path[0]+'\\..\\')

from utils import config
from nets import network as nn

_config_ = config.config()
_config_._print()
config.__show_details__()

model = nn.build_model(_config_)

model.summary()

print('no of layers ', len(model.layers))