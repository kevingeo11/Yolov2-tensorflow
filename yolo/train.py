import numpy as np
import tensorflow as tf
from os.path import join
import matplotlib.pyplot as plt

from utils import config
from utils.data_loader import DataGenerator
from utils.loss import custom_loss
from nets import network as nn

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        if config.RUNCPU:
            print('GPU device not found')
        else:
            raise SystemError('GPU device not found')
    else:    
        print('Found GPU at: {}'.format(device_name))

    _config_ = config.config()
    _config_._print()
    config.__show_details__()

    train_file = join(config.base_dir, 'train.txt')
    val_file = join(config.base_dir, 'val.txt')

    img_path = config.image_dir
    label_path = config.label_dir

    with open(train_file, 'r') as f:
        train_list = f.readlines()
    train_list = [item.replace('\n','') for item in train_list]
    print('num of train images ', len(train_list))

    with open(val_file, 'r') as f:
        val_list = f.readlines()
    val_list = [item.replace('\n','') for item in val_list]
    print('num of val images ', len(val_list))

    train_batch_generator = DataGenerator(train_list, _config_, img_path, label_path)
    val_batch_generator = DataGenerator(val_list, _config_, img_path, label_path)

    model = nn.build_model(_config_)
    if config.load_weights:
        weight_loc = join(config.weights_dir, config.load_model_name)
        model.load_weights(weight_loc)

    early_stop = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(join(config.weights_dir, config.model_name), save_weights_only=True, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = ModelCheckpoint(join(config.weights_dir, config.model_name_val), save_weights_only=True,
                                          monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    optimizer = Adam(learning_rate = config.lr)
    model.compile(loss = custom_loss(_config_) , optimizer = optimizer)

    callbacks = [early_stop, checkpoint, checkpoint_acc]

    history = model.fit(x = train_batch_generator, steps_per_epoch = len(train_batch_generator), epochs = config.epoch, 
                        verbose = 1, callbacks = callbacks, 
                        validation_data = val_batch_generator, validation_steps = len(val_batch_generator))

    plt.figure(figsize = (7,7))
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='val')
    plt.legend()
    plt.show()

    plt.figure(figsize = (7,7))
    plt.plot(history.history['loss'], 'r', label='train')
    plt.legend()
    plt.show()

    plt.figure(figsize = (7,7))
    plt.plot(history.history['val_loss'], 'b', label='val')
    plt.legend()
    plt.show()
    