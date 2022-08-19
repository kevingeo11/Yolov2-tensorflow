import numpy as np
import tensorflow as tf
from os.path import join
import cv2

from utils.config import config


class Box:
    def __init__(self, x, y, w, h, confidence=None, classes=None):
        self.x, self.y = x, y
        self.w, self.h = w, h
        ## the code below are used during inference
        # probability
        self.confidence = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)
        
        self.xmin = x - w/2
        self.xmax = x + w/2
        self.ymin = y - h/2
        self.ymax = y + h/2
        
    def set_class(self,classes):
        self.classes = classes
        self.label = np.argmax(self.classes) 
        
    def get_label(self):  
        return self.label
    
    def get_score(self):
        return self.classes[self.label]

class OutputRescaler(object):
    def __init__(self, config):
        self.anchors = config.anchors

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        
        return e_x / e_x.sum(axis, keepdims=True)
    
    def get_shifting_matrix(self,netout):
        
        grid_h, grid_w, n_anchors = netout.shape[:3]
        no = netout[...,0]
        
        anchors_w = self.anchors[:,0]
        anchors_h = self.anchors[:,1]
       
        mat_grid_w = np.zeros_like(no)
        for igrid_w in range(grid_w):
            mat_grid_w[:,igrid_w,:] = igrid_w

        mat_grid_h = np.zeros_like(no)
        for igrid_h in range(grid_h):
            mat_grid_h[igrid_h,:,:] = igrid_h

        mat_anchor_w = np.zeros_like(no)
        for ianchor in range(n_anchors):    
            mat_anchor_w[:,:,ianchor] = anchors_w[ianchor]

        mat_anchor_h = np.zeros_like(no) 
        for ianchor in range(n_anchors):    
            mat_anchor_h[:,:,ianchor] = anchors_h[ianchor]
            
        return mat_grid_w, mat_grid_h, mat_anchor_w, mat_anchor_h

    def fit(self, netout):    

        grid_h, grid_w, n_anchors = netout.shape[:3]
        
        mat_grid_w, mat_grid_h, mat_anchor_w, mat_anchor_h = self.get_shifting_matrix(netout)

        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_grid_w)/grid_w 
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_grid_h)/grid_h 
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_anchor_w)
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_anchor_h)

        netout[..., 4]   = self._sigmoid(netout[..., 4])

        expand_conf      = np.expand_dims(netout[..., 4], -1)
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:]) 
    
        return netout 


class inference():
    def __init__(self, config) -> None:
        self.config = config
        self.model = None

    def set_model(self, nn):
        self.model = nn.build_model(self.config)

    def set_weights(self, loc):
        if self.model is None:
            print('Model not set - call set_model first')
            return
        self.model.load_weights(loc)

    def __encode_core(self, image, rgb=True, norm=True):
        image = cv2.resize(image, (self.config.h, self.config.w))
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if norm:
            image = self.__normalize(image)
            
        return image

    def __find_high_class_probability_bbox(netout_scale, obj_threshold):

        grid_h, grid_w, n_anchors = netout_scale.shape[:3]
        
        boxes = []
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(n_anchors):
                    # from 4th element onwards are confidence and class classes
                    classes = netout_scale[row, col, b, 5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout_scale[row, col, b,:4]
                        confidence = netout_scale[row, col, b, 4]
                        box = Box(x, y, w, h, confidence, classes)
                        if box.get_score() > obj_threshold:
                            boxes.append(box)
        return boxes

    def __normalize(image):
        return image / 255.

    def run(self, img_loc):
        if img_loc is None:
            print('specify location')
            return
        
        image = cv2.imread(img_loc)
        if image is None:
            print('Cannot find Image - check location')
            return

        image = self.__encode_core(image)
        x_input = np.expand_dims(image, 0)
        ypred = self.model.predict(x_input)

        return ypred

    def converter_single(self, ypred):
        outputRescaler = OutputRescaler(self.config)
        output_scaled = outputRescaler.fit(ypred)

        obj_threshold = 0.03
        boxes = self.__find_high_class_probability_bbox(output_scaled,obj_threshold)

        return boxes




    