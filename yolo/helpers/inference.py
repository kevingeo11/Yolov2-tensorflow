import numpy as np
import tensorflow as tf
from os.path import join
import cv2
import seaborn as sns
import copy

class BestAnchorBoxFinder(object):
    def __init__(self, config):
        '''
            self.anchors : list of all anchor boxes
                           Each element in the list is a Box class
                           For more information please see Box class
        '''
        self.anchors = [Box(config.anchors[i][0]/2, config.anchors[i][1]/2, config.anchors[i][0], config.anchors[i][1]) 
                        for i in range(len(config.anchors))]
        
    def __interval_overlap(self, interval_a, interval_b):
        '''
            Calculates the interval between two points in one dimensions

            .__________.      x3        x4
            x1         x2     ._________.
            In the above case interval = 0

            x1          x2
            .___________.
                 .________________.
                 x3               x4
            In the above case interval = x2 - x3

            x1 : min of box1
            x3 : min of box2
            x2 : max of box1
            x4 : max of box2

            Returns the interval

        '''
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self, box1, box2):
        '''
            Calculates intersection over union of 2 boxes
            .______________.
            |              |
            |       intersect_w
            |     .________|_____.
            |     |/ / / / |     |          Intersect area = intersect_w * intersect_h
            |     |/ / / / | intersect_h    Union = Area of Box1 + Area of Box2 - Intersect Area
            ._____|________.     |          
                  |              |          Returns the IoU
                  |              |
                  |              |
                  .______________.

        '''
        intersect_w = self.__interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.__interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union

    ## below function curretly not used
    
    # def find(self, center_w, center_h):
    #     # find the anchor that best predicts this box
    #     best_anchor = -1
    #     max_iou     = -1
    #     # each Anchor box is specialized to have a certain shape.
    #     # e.g., flat large rectangle, or small square
    #     shifted_box = Box(center_w/2, center_h/2,center_w, center_h)
    #     ##  For given object, find the best anchor box!
    #     for i in range(len(self.anchors)): ## run through each anchor box
    #         anchor = self.anchors[i]
    #         iou    = self.__bbox_iou(shifted_box, anchor)
    #         if max_iou < iou:
    #             best_anchor = i
    #             max_iou     = iou
    #     return(best_anchor,max_iou) 

class Box:
    '''
        This class defines a box which is used for prediction
    '''
    def __init__(self, x, y, w, h, confidence = None, classes = None):
        '''
            x : x coordinate for center of the box
            y : y coordinate for center of the box
            w : width of the box
            h : height of the box
        '''
        self.x, self.y = x, y
        self.w, self.h = w, h

        self.confidence = confidence
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

    def __find_high_class_probability_bbox(self, netout_scale, obj_threshold):

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

    def __normalize(self, image):
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

        outputRescaler = OutputRescaler(self.config)
        output_scaled = outputRescaler.fit(ypred[0])

        obj_threshold = 0.03
        boxes = self.__find_high_class_probability_bbox(output_scaled,obj_threshold)

        iou_threshold = 0.01
        final_boxes = self.nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)

        ima = self.draw_boxes(image,final_boxes,self.config.classes,verbose=True)

        return ima
        

    def draw_boxes(self, image, boxes, labels, obj_baseline=0.05,verbose=False):
        '''
        image : np.array of shape (N height, N width, 3)
        '''
        def adjust_minmax(c,_max):
            if c < 0:
                c = 0   
            if c > _max:
                c = _max
            return c
        
        image = copy.deepcopy(image)
        image_h, image_w, _ = image.shape
        score_rescaled  = np.array([box.get_score() for box in boxes])
        score_rescaled /= obj_baseline
        
        colors = sns.color_palette("husl", 8)
        for sr, box,color in zip(score_rescaled,boxes, colors):
            xmin = adjust_minmax(int(box.xmin*image_w),image_w)
            ymin = adjust_minmax(int(box.ymin*image_h),image_h)
            xmax = adjust_minmax(int(box.xmax*image_w),image_w)
            ymax = adjust_minmax(int(box.ymax*image_h),image_h)
    
            
            text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())
            if verbose:
                print("{} xmin={:4.0f},ymin={:4.0f},xmax={:4.0f},ymax={:4.0f}".format(text,xmin,ymin,xmax,ymax,text))
            cv2.rectangle(image, 
                        pt1=(xmin,ymin), 
                        pt2=(xmax,ymax), 
                        color=color, 
                        thickness=5)
            cv2.putText(img       = image, 
                        text      = text, 
                        org       = (xmin+ 13, ymin + 13),
                        fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1e-3 * image_h,
                        color     = (1, 0, 1),
                        thickness = 1)
            
        return image

    def nonmax_suppression(self, boxes, iou_threshold, obj_threshold):
        '''
        boxes : list containing "good" BoundBox of a frame
                [BoundBox(),BoundBox(),...]
        '''
        bestAnchorBoxFinder = BestAnchorBoxFinder(self.config)
        
        CLASS = len(boxes[0].classes)
        index_boxes = []   
        # suppress non-maximal boxes
        for c in range(CLASS):
            # extract class probabilities of the c^th class from multiple bbox
            class_probability_from_bbxs = [box.classes[c] for box in boxes]

            #sorted_indices[i] contains the i^th largest class probabilities
            sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                # if class probability is zero then ignore
                if boxes[index_i].classes[c] == 0:  
                    continue
                else:
                    index_boxes.append(index_i)
                    for j in range(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                        # if so, the remaining bbox' class probabilities are set to 0.
                        bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                        if bbox_iou >= iou_threshold:
                            classes = boxes[index_j].classes
                            classes[c] = 0
                            boxes[index_j].set_class(classes)
                            
        newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]                
        
        return newboxes