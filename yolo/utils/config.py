import numpy as np
import sys
from os.path import join

'''
    Anchor boxes
    ---------------
    Anchor boxes are defined for easier prediction.
    Each grid cell can have n number of anchor boxes.
    anchor box is defined as [bw, bh] where bw and bh are the width and height respectively and scaled to [0,1]
    To calculate the appropriate anchor box dimensions for your application - Run kmeans clustering algorithm on your training data. 
'''
__anchors__ = np.array([[0.06602624, 0.0513364 ],
                        [0.09009859, 0.06823154],
                        [0.12998022, 0.09910336]])

class config(object):
    def __init__(self,):
        self.n_anchors = 3
        self.anchors = __anchors__
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.n_classes = len(self.classes)
        self.w = 256
        self.h = 256
        '''
            grid sizes come from the output of the conv network
            check model.summary() for grid size values
        '''
        self.grid_w = 8
        self.grid_h = 8
        '''
            batch_size should be a perfect divisor of train and val length
        '''
        self.batch_size = 8
        self.l2_coeff = 0.001
    
    def _print(self):
        print('*'*10 + ' config ' + '*'*10)
        print('number of anchors: ', self.n_anchors)
        print('anchor boxes: ', self.anchors)
        print('number if anchor boxes: ', self.n_classes)
        print('list of classes: ', self.classes)
        print('width: ', self.w)
        print('height: ', self.h)
        print('grid w: ', self.grid_w)
        print('grid h: ', self.grid_h)
        print('batch size: ', self.batch_size)
        print('l2_coeff : ', self.l2_coeff)
        print('*'*10 + '*'*8 + '*'*10)

'''
    directory in windows and ubuntu format
    else block will take of linux format in colab
'''

if sys.platform == 'win32':
    base_dir = '..//PASCAL Data'
    image_dir = '..//PASCAL Data\\Images'
    label_dir = '..//PASCAL Data\\Labels'
else:
    base_dir = '../PASCAL Data'
    image_dir = '../PASCAL Data/Images'
    label_dir = '../PASCAL Data/Labels'

weights_dir = 'weights'
load_weights = False
load_model_name = '.h5'

model_name = 'pascal.h5'
model_name_val = 'pascal_val.h5'

epoch = 10
lr = 0.0001

RUNCPU = True

def __show_details__():
    print('*'*10 + ' details ' + '*'*10)
    print('Platform : ', sys.platform)
    print('base_dir : ', base_dir)
    print('image_dir : ', image_dir)
    print('label_dir : ', label_dir)
    print('weights_dir : ', weights_dir)
    print('load_weights : ', load_weights)
    print('load_model_name : ', load_model_name)
    print('model_name : ', model_name)
    print('model_name_val : ', model_name_val)
    print('epoch : ', epoch)
    print('lr : ', lr)
    print('*'*10 + '*'*9 + '*'*10)
