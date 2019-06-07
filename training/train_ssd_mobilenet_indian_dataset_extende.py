import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from models.ssd_mobilenet_extended import ssd_300
from misc.keras_ssd_loss import SSDLoss, FocalLoss, weightedSSDLoss, weightedFocalLoss
from misc.keras_layer_AnchorBoxes import AnchorBoxes
from misc.keras_layer_L2Normalization import L2Normalization
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from misc.ssd_batch_generator import BatchGenerator
from keras.utils.training_utils import multi_gpu_model
import os
import keras

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
subtract_mean = [127.5,127.5,127.5]  # The per-channel mean of the images in the dataset
swap_channels = True  # The color channel order in the original SSD is BGR
n_classes = 2  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = [0.07,0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]

# scales = scales_voc
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]  # The anchor box aspect ratios used in the original SSD300; the order matters

two_boxes_for_ar1 = True
steps = [8,16, 32, 64, 100, 150, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5,0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

# K.clear_session()  # Clear previous models from memory.

model = ssd_300("training",
                image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                limit_boxes=limit_boxes,
                variances=variances,
                coords=coords,
                normalize_coords=normalize_coords,
                subtract_mean=subtract_mean,
                divide_by_stddev=127.5,
                swap_channels=swap_channels)



for layer in model.layers:
    layer.name = layer.name + "_v1"


#model.load_weights("/home/manish/MobileNet-ssd-keras/conversion/converted_model.h5",by_name = True, skip_mismatch = True)
model.load_weights("/media/shareit/weights/ssd300_mobilenet_city_map/ssd300_epoch-52.h5" ,by_name = True, skip_mismatch = True)

predictor_sizes = [model.get_layer('conv3_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv11_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv13_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv14_2_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv15_2_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv16_2_mbox_conf_v1').output_shape[1:3],
                   model.get_layer('conv17_2_mbox_conf_v1').output_shape[1:3]]




adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = FocalLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

#ssd_loss = weightedFocalLoss(weights=class_weights)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)



train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

VOC_2007_images_dir = '/media/shareit/VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_dir = '/media/shareit/VOCdevkit/VOC2012/JPEGImages/'

# The directories that contain the annotations.
VOC_2007_annotations_dir = '/media/shareit/VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_dir = '/media/shareit/VOCdevkit/VOC2012/Annotations/'

train_hitech_file = "/media/shared1/ssd_data/detection_data/train.txt"
valid_hitech_file = "/media/shared1/ssd_data/detection_data/val.txt"


# The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor']
# # classes_hitech = ['background','A/C', 'B/M', 'B/T', 'P', 'Cow']
# map_dict = {0:1, 1:2, 2:3, 3:4, 4:5}

classes_hitech = ['background','A/C', 'P']
map_dict = {0:1, 1:2, 2:1, 3:2, 4:2}




train_dataset.parse_darknet(images_dirs=[VOC_2007_images_dir,
                                         VOC_2012_images_dir],
                            image_set_filenames=[train_hitech_file],
                            annotations_dirs=[VOC_2007_annotations_dir,
                                              VOC_2012_annotations_dir],
                            classes=classes_hitech,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False,
                            map_dict = map_dict, 
                            label_string="ssd_labels")



val_dataset.parse_darknet(images_dirs=[VOC_2007_images_dir,
                                       VOC_2012_images_dir],
                          image_set_filenames=[valid_hitech_file],
                          annotations_dirs=[VOC_2007_annotations_dir,
                                            VOC_2012_annotations_dir],
                          classes=classes_hitech,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False,
                          map_dict = map_dict, 
                          label_string="ssd_labels")



# 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                steps=steps,
                                offsets=offsets,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 4: Set the batch size.

batch_size = 32

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         convert_to_3_channels=True,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.4)

val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         convert_to_3_channels=True,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                         # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.4)
# Get the number of samples in the training and validations datasets to compute the epoch lengths below.
n_train_samples = train_dataset.get_n_samples()
n_val_samples = val_dataset.get_n_samples()



def lr_schedule(epoch):
    if epoch <= 200:
        return 0.001
    elif epoch <= 400:
        return 0.0005
    else:
        return 0.0001


learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
# learning_rate_scheduler.set_model(org_model)

checkpoint = ModelCheckpoint('/media/shareit/weights/ssd300_mobilenet_inidan_Datset_extended/ssd300_epoch-{epoch:02d}.h5')

# terminate_on_nan = TerminateOnNaN()

tensorborad = TensorBoard(log_dir='/media/shareit/weights/ssd300_mobilenet_inidan_Datset_extended/logs',
                          histogram_freq=0, write_graph=True, write_images=True)

# tensorborad.set_model(org_model)

# cbk = MyCbk(org_model)


callbacks = [checkpoint,tensorborad,learning_rate_scheduler]

# TODO: Set the number of epochs to train for.
epochs = 500
intial_epoch = 0

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(n_train_samples)/batch_size,
                              verbose=1,
                              initial_epoch=intial_epoch,
                              epochs=epochs,
                              validation_data=val_generator,
                              validation_steps=ceil(n_val_samples)/batch_size,
                              callbacks=callbacks
                              )
