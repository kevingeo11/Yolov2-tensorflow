# YoloV2 Implemented in TensorFlow 2.0

This repo provides an implementation of YoloV2 in TensorFlow 2.0.

## Usage

### Detection

```bash
# yolov3
python detect.py --image ./data/meme.jpg

# yolov3-tiny
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

# video file with output
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```

### Training

I have created a complete tutorial on how to train from scratch using the VOC2012 Dataset.
See the documentation here https://github.com/zzh8829/yolov3-tf2/blob/master/docs/training_voc.md

For customzied training, you need to generate tfrecord following the TensorFlow Object Detection API.
For example you can use [Microsoft VOTT](https://github.com/Microsoft/VoTT) to generate such dataset.
You can also use this [script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py) to create the pascal voc dataset.

Example commend line arguments for training
``` bash
python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode eager_tf --transfer fine_tune

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode fit --transfer none

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 100 --mode fit --transfer no_output

python train.py --batch_size 8 --dataset ~/Data/voc2012.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 10 --mode eager_fit --transfer fine_tune --weights ./checkpoints/yolov3-tiny.tf --tiny
```

## References

It is pretty much impossible to implement this from the yolov3 paper alone. I had to reference the official (very hard to understand) and many un-official (many minor errors) repos to piece together the complete picture.

- https://github.com/pjreddie/darknet
    - official yolov3 implementation
- https://github.com/AlexeyAB
    - explinations of parameters
- https://github.com/qqwweee/keras-yolo3
    - models
    - loss functions
- https://github.com/YunYang1994/tensorflow-yolov3
    - data transformations
    - loss functions
- https://github.com/ayooshkathuria/pytorch-yolo-v3
    - models
- https://github.com/broadinstitute/keras-resnet
    - batch normalization fix
