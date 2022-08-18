import tensorflow as tf
import numpy as np

def get_cell_grid(grid_w, grid_h, batch_size, n_anchor): 
    '''
        Helper function to assure that the bounding box x and y are in the grid cell scale
        -- output --
        output[i, 5, 3, :, :] = array([[3., 5.],
                                    [3., 5.],
                                    [3., 5.]], dtype=float32)
        where i varies from 0 to batch_size - 1
    '''

    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, n_anchor, 1])
    
    return cell_grid

def scale_prediction(y_pred, cell_grid, anchors, n_anchors, anchor_scale = None):    
    '''
        Scale prediction vector - y_pred
        
        -- input --
        
        y_pred : takes any real values
                 tensor of shape = (batch_size, grid_h, grid_w, n_anchor, 4 + 1 + n_classes)
        cell_grid : contains the value of the current grid in terms of (x, y)
                    tensor of shape = (batch_size, grid_h, grid_w, n_anchor, 2)
                    e.g. cell_grid[batch, 5, 6, 0, :] = array([6., 5.], dtype=float32)
                    output of get_cell_grid function
        anchors : anchor boxes in [0, 1] scale
                  shape = (n_anchor, 2)
        n_anchors : number of anchor boxes defined

        -- output --
        
        pred_box_xy : tensor of shape = (batch_size, grid_h, grid_w, n_anchor, 2), contianing [center_y, center_x]  
                      [center_x, center_y] ranges from [0, 0] to [grid_w-1, grid_h-1]
                      pred_box_xy[irow, igrid_h, igrid_w, ianchor, 0] =  center_x
                      pred_box_xy[irow, igrid_h, igrid_w, ianchor, 1] =  center_y
                                                   
        pred_Box_wh : tensor of shape = (batch_size, grid_h, grid_w, n_anchor, 2), containing [width and height]
                      [width and height] ranges from [0, 0] to [grid_w, grid_h]
                      warning : currently there is a scale issue due to calculation
        
        pred_box_conf : tensor of shape = (batch_size, grid_h, grid_w, n_anchor, 1), containing confidence to range between 0 and 1
        
        pred_box_class : tensor of shape = (batch_size, grid_h, grid_w, n_anchor, n_classes) containing class probabilities
                         tf.nn.sparse_softmax_cross_entropy_with_logits expects unscaled logits, since it performs a softmax on logits internally for efficiency.
                         Therefore we are keeping pred_box_classes unscaled.

        -- Calculation --

        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = Pw * exp(tw)
        bh = Ph * exp(th)
        Pb = sigmoid(tb)
    '''
    
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    if anchor_scale is not None:
        scaled_anchors = anchors*anchor_scale
    else:
        scaled_anchors = anchors
    
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(scaled_anchors, [1,1,1,n_anchors,2])

    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    pred_box_class = tf.sigmoid(y_pred[..., 5:])
    
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def extract_ground_truth(y_true):    
    '''
        Extracting true label to xy, wh, confidence and class labels.
        true_box_class : index of the seen class is recorded

        true_box_xy : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 2)
        true_box_wh : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 2)
        true_box_conf : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
        true_box_class : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
    '''
    true_box_xy    = y_true[..., 0:2] 
    true_box_wh    = y_true[..., 2:4]
    true_box_conf  = y_true[...,4]
    true_box_class = tf.math.argmax(y_true[..., 5:], -1)

    return true_box_xy, true_box_wh, true_box_conf, true_box_class

def calc_loss_xywh(true_box_conf, lambda_coord, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):  
    '''
        calculate the xywh loss
        loss_xywh = (lambda_coord/n_obj) * {sum across grid and anchors}(L_{i,j}_obj)[(xt - xp)**2 + (yt - yp)**2 + 
                                                                                      (sqrt(wt) - sqrt(wp))**2 + (sqrt(ht) - sqrt(hp))**2]
        
        coord_mask : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 1)
                     1 in (grid, anchor) pair where P is 1 or object exists

        -- input --    
            true_box_conf : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
            true_box_xy : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 2)
            pred_box_xy : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 2)
            pred_box_wh : tensor of shape (batch_size, grid_h, grid_w, n_anchor, 2)
       
    '''
    
    coord_mask = tf.expand_dims(true_box_conf, axis = -1) * lambda_coord 
    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask)
    loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh)) * coord_mask)
    loss_xywh = (loss_xy + loss_wh) / (nb_coord_box + 1e-6)

    return loss_xywh

def calc_loss_class(true_box_conf, lambda_class, true_box_class, pred_box_class):
    '''
        calculate the xywh loss
        loss_class = -(lambda_class/n_obj) * {sum across grid and anchors}(L_{i,j}_obj){sum across classes}(P_t * log(P_p))
        
        class_mask : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
                     1 in (grid, anchor) pair where P is 1 or object exists
        
        -- input --         
            true_box_conf : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
            true_box_class :tensor of shape (batch_size, grid_h, grid_w, n_anchor) contains class index
            pred_box_class : tensor of shape (batch_size, grid_h, grid_w, n_anchor, n_classes)
    
    ''' 

    class_mask = true_box_conf * lambda_class
    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))

    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class, 
                                                                  logits = pred_box_class)

    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6) 
    
    return loss_class

def calc_loss_conf(true_box_conf, pred_box_conf, lambda_obj, lambda_noobj, n_conf):  
    '''
        calculate the xywh loss
        loss_conf = (lambda_obj/n_conf) * {sum across grid and anchors}(L_{i,j}_obj)(1 - Pc_p)**2 + 
                    (lambda_noobj/n_conf) * {sum across grid and anchors}(L_{i,j}_noobj)(0 - Pc_p)**2 
                    where n_conf = {sum across grid and anchors}(1)
        
        obj_mask : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
                   1 in (grid, anchor) pair where P is 1 or object exists

        noobj_mask : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
                     1 in (grid, anchor) pair where P is 0 or object dosen't exists

        -- input --
            true_box_conf : tensor of shape (batch_size, grid_h, grid_w, n_anchor)
            pred_box_conf : tensor of shape (batch_size, grid_h, grid_w, n_anchor)

    '''

    obj_mask = true_box_conf * lambda_obj
    noobj_mask = (1 - true_box_conf) * lambda_noobj
    
    noobj_loss = noobj_mask * tf.square(0 - pred_box_conf)
    obj_loss = obj_mask * tf.square(1 - pred_box_conf)

    loss_conf = (tf.reduce_sum(noobj_loss) + tf.reduce_sum(obj_loss)) / n_conf
    
    return loss_conf


def custom_loss(config__):
    grid_w = config__.grid_w
    grid_h = config__.grid_h
    n_anchors = config__.n_anchors
    batch_size = config__.batch_size
    anchors = config__.anchors

    def loss_fun(y_true, y_pred):
        '''
            y_true : (n batch_size, n grid_h, n grid_w, n_anchor, 4 + 1 + n_classes)
                y_true[irow, i_gridh, i_gridw, i_anchor, :4] = center_x, center_y, w, h
                    center_x : The x coordinate center of the bounding box.
                            Rescaled to range between 0 and gird_w e.g. ranging between [0, 13]
                    center_y : The y coordinate center of the bounding box.
                            Rescaled to range between 0 and gird_h e.g., ranging between [0, 13]
                    w        : The width of the bounding box.
                            Rescaled to range between 0 and gird_w e.g., ranging between [0, 13]
                    h        : The height of the bounding box.
                            Rescaled to range between 0 and gird_h e.g., ranging between [0, 13]
                            
                y_true[irow, i_gridh, i_gridw, i_anchor, 4] = ground truth confidence --> in range [0, 1]
                    
                    ground truth confidence is 1 if object exists in this (anchor box, gird cell) pair else 0
                
                y_true[irow, i_gridh, i_gridw, i_anchor, 5 + iclass] = 1 if the object is in category  else 0
            
            y_pred : (n batch_size, n grid_h, n grid_w, n_anchor, 4 + 1 + n_classes)
                y_pred[irow, i_gridh, i_gridw, i_anchor, :4] = center_x, center_y, w, h
                    center_x : The x coordinate center of the bounding box. Value can be any real number.
                    center_y : The y coordinate center of the bounding box. Value can be any real number.
                    w        : The width of the bounding box. Value can be any real number.
                    h        : The height of the bounding box. Value can be any real number.
                            
                y_pred[irow, i_gridh, i_gridw, i_anchor, 4] = ground truth confidence --> Value can be any real number.
                y_pred[irow, i_gridh, i_gridw, i_anchor, 5 + iclass] = class label --> Value can be any real number.
        '''

        cell_grid = get_cell_grid(grid_w, grid_h, batch_size, n_anchors)

        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = scale_prediction(y_pred, cell_grid, anchors, 
                                                                                    n_anchors, anchor_scale = np.array([grid_w, grid_h]))
        true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)

        lamdda_coord = 1
        loss_xywh = calc_loss_xywh(true_box_conf, lamdda_coord, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)

        lambda_class = 1
        loss_class  = calc_loss_class(true_box_conf, lambda_class, true_box_class, pred_box_class)

        lambda_obj = 1
        lambda_noobj = 0.5
        n_conf = grid_w * grid_h *n_anchors
        loss_conf = calc_loss_conf(true_box_conf, pred_box_conf, lambda_obj, lambda_noobj, n_conf)
        
        loss = loss_xywh + loss_conf + loss_class

        return loss
    
    return loss_fun