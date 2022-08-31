# YoloV2 Implemented in TensorFlow 2.0

This repo provides an implementation of YoloV2 from scratch in TensorFlow 2.0.

Note: The current config.py is set up for the PASCAL Dataset. Please adjust the parameters in config.py according to your specifications.

## Usage

### Setting up Config

Setup the config file located in yolo/utils/
Details like anchor box sizes, width and height of the image, learning rate can be set in config.py
Training and Inferance scripts use config.py to run properly

### Detection

Run the Interpreter Notebook inside yolo folder.

Loading config
```bash
from utils import config
_config_ = config.config()
```

Loading model
```bash
from nets import network as nn
from tensorflow.keras.models import load_model

model = nn.build_model(_config_)
weight_loc = 'location of where weights are stored'
model.load_weights(weight_loc)
```

Running Inference code
```bash
from helpers.inference import inference

inference = inference(_config_, model)

img_loc = 'location of the image'
ypred = inference.run(img_loc)

## ypred returns the image after drawing the bounding box for the detected objects
```

### Training

Update the config.py file and run train.py
```bash
python train.py
```

## References

- https://github.com/pjreddie/darknet

- https://github.com/zzh8829/yolov3-tf2

- https://github.com/YunYang1994/tensorflow-yolov3