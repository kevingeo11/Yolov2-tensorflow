import tensorflow as tf
import numpy as np
import cv2
from os.path import join

class BestAnchorBoxFinder(object):
    '''
        Object to find the best anchor box that has the max iou with the given width and height.
        self.anchors : numpy array that contains all anchor boxes
                       shape of (n_anchors, 2)
        self.bbox_iou : calculates the IOU between box1 and box2, box contains [w, h] where w and h are in the range [0, 1]
                        .____________.______.
                        |     w1     |  w2  |
                        |            |      | h2
                     h1 |____________|______.
                        |   box1     |      box2
                        .____________.
        self.find : calculates the ious between the given boxes and the list of anchor boxes and returns the max iou anchor box
                    input - center_w and center_h, width and height of given box, both in range [0, 1]
    '''
    def __init__(self, anchors):
        self.anchors = anchors

    def bbox_iou(self, box1, box2):
        w = min(box1[0], box2[0])
        h = min(box1[1], box2[1])
        intersect = w*h
        
        union = box1[0]*box1[1] + box2[0]*box2[1] - intersect

        return float(intersect) / union
    
    def find(self, center_w, center_h):
        '''
            -- input --
            center_w : width of the input box in [0, 1]
            center_h : height of the input box in [0, 1]

            -- output --
            best_anchor : index of the best anchor box in the self.anchors array
            max_iou : iou of the best anchor box
        '''
        best_anchor = -1
        max_iou = -1

        for idx, anchor in zip(range(len(self.anchors)), self.anchors):
            iou = self.bbox_iou([center_w, center_h], anchor)
            if max_iou < iou:
                best_anchor = idx
                max_iou = iou
        return best_anchor, max_iou

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, config, img_path, label_path, shuffle = True):
        '''
            self.config : config object - for more information refer utils/config.py
            self.img_path : (str) path where images are stored
            self.label_path : (str) path where labels are stored
            self.fnames : (List) List of image names for the generater to read from.
                          elements (str) in this list do not have '.jpg' or '.txt' extensions
            self.batch_size : (int) batch size for reading data
            self.bestAnchorBoxFinder : BestAnchorBoxFinder object for finding max iou anchor box
            self.shuffle : shuffle hyperparameter. if true dataset is shuffled after every epoch
        '''
        self.config = config
        self.img_path = img_path
        self.label_path = label_path
        self.fnames = fnames
        self.batch_size = self.config.batch_size
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config.anchors)
        self.shuffle = shuffle
        if self.shuffle: 
            np.random.shuffle(self.fnames)
    
    def norm(self, image):
        return image / 255.
    
    def __len__(self):
        return int(np.ceil(float(len(self.fnames))/self.config.batch_size))
    
    def __getitem__(self, index):
        '''
            -- output --
            X : shape (batch_size, image_size, image_size, 3)
            Y : shape (batch_size, grid_h, grid_w, n_anchors, 4 + 1 + n_classes)

        '''
        
        image_batch = [item + '.jpg' for item in self.fnames[index * self.batch_size:(index + 1) * self.batch_size]]
        label_batch = [item + '.txt' for item in self.fnames[index * self.batch_size:(index + 1) * self.batch_size]]

        X, Y = self.__get_data(image_batch, label_batch)        
        
        return X, Y
    
    def __get_data(self, image_batch, label_batch):
        X_batch = np.asarray([self.__read_image(name) for name in image_batch])
        Y_batch = np.asarray([self.__read_label(name) for name in label_batch])
        
        return X_batch, Y_batch
    
    def __read_image(self, name):
        '''
            output : Images are in the RGB colour space and are normalised to [0, 1]
        '''
        img = cv2.imread(join(self.img_path, name))
        img = cv2.resize(img, (self.config.h, self.config.w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return self.norm(img)
    
    def __read_label(self, name):
        '''
            -- output --

            label_matrix : Label or y_true for an object/item
                           shape : (grid_h, grid_w, n_anchors, 4+1+n_classes)
                           label_matrix[igrid_h, igrid_w, ianchor] contains n_anchors anchor boxes.
                                self.bestAnchorBoxFinder is used to find which anchor box the true label should belong to.
                           label_matrix[igrid_h, igrid_w, ianchor] = [x, y, w, h, P, classes]
                                x : x coord of center of the true label box. range [0, grid_w]
                                y : y coord of center of the true label box. range [0, grid_h]
                                w : width of true label box. range [0, grid_w]
                                h : height of true label box. range [0, grid_h]
                                P : probability if object exist == 1 (for true label)
                                label_matrix[igrid_h, igrid_w, ianchor, 5:] : one hot vector for class labels
                                    i.e. label_matrix[igrid_h, igrid_w, ianchor, 5 + iclass] = 1 if iclass'th class object exist
                                    in (igrid, ianchor) pair

        '''

        with open(join(self.label_path, name), 'r') as f:
            annotations = f.readlines()
            
        label_matrix = np.zeros([self.config.grid_h, self.config.grid_w, self.config.n_anchors, 4 + 1 + self.config.n_classes])
        
        for annotation in annotations:
            cls, x, y, w, h = [float(item) if float(item) != int(float(item)) else int(item) for item in annotation.replace('\n','').split(' ')]
            loc_x, loc_y = self.config.grid_w*x, self.config.grid_h*y
            grid_x = int(np.floor(loc_x))
            grid_y = int(np.floor(loc_y))
            
            box_w, box_h = self.config.grid_w*w, self.config.grid_h*h
            
            box = [loc_x, loc_y, box_w, box_h]
            
            best_anchor, max_iou = self.bestAnchorBoxFinder.find(w, h)
            
            label_matrix[grid_y, grid_x, best_anchor, :4] = box
            label_matrix[grid_y, grid_x, best_anchor, 4] = 1.
            label_matrix[grid_y, grid_x, best_anchor, 5 + cls] = 1
            
        return label_matrix

    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.fnames)