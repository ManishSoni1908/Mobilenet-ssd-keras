import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
import keras
import caffe
import cv2
import numpy as np 
from models.ssd_mobilenet import ssd_300
import math
import os
import shutil
import stat
import subprocess
import sys
import cv2
import numpy as np
import caffe
from google.protobuf import text_format
from keras.optimizers import Adam
from misc.keras_ssd_loss import SSDLoss
# from misc.
import os
import h5py
import keras
from keras.preprocessing import image
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

weights_file = "/Path to /MobileNet-SSD/mobilenet_iter_73000.caffemodel"
deploy_file  = "/Path to /MobileNet-SSD/infer.prototxt"


caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(deploy_file, weights_file,caffe.TEST)


print "done"


img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
subtract_mean = None  # The per-channel mean of the images in the dataset
swap_channels = False  # The color channel order in the original SSD is BGR
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]
aspect_ratios = [[1.00001, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [16, 32, 64, 100, 150, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

# K.clear_session()  # Clear previous models from memory.

model = ssd_300("inference",
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
                divide_by_stddev=None,
                swap_channels=swap_channels)



# model.save("my_model.h5")

# # model.load_weights("/home/manish/ssd-cutomize-manish/pre_trained_model/mobilenet.h5",skip_mismatch=True,by_name = True)

for layer in model.layers:
    layer_type = type(layer).__name__
    name = layer.name

    print name

    if layer_type=='Conv2D' or layer_type=='Convolution2D':
        l = []

        # print "Caffe ", net.params[name][0].data.shape
        # print "Keras ", model.get_layer("conv0").get_weights()[0].shape
        l.append(net.params[name][0].data.transpose(2,3,1,0))

        # print name
        if(len(net.params[name]) >1):
            l.append(net.params[name][1].data)
            model.get_layer(name).set_weights(l)
            print "bias avilable for the layer " ,name

        else:
            model.get_layer(name).set_weights(l)
            print "No bias for ",name


    if layer_type=='DepthwiseConv2D':
        l = []
        l.append(net.params[name][0].data.transpose(2,3,0,1))

        # print name
        if(len(net.params[name]) >1):
            l.append(net.params[name][1].data)
            model.get_layer(name).set_weights(l)
            print "bias avilable for the layer " ,name

        else:
            model.get_layer(name).set_weights(l)
            print "No bias for ",name


    if layer_type=='BatchNormalization':
        l = []

        scale_name = name.replace("bn","scale")
      
        l.append(np.array(net.params[scale_name][0].data))
        l.append(np.array(net.params[scale_name][1].data))
		l.append(np.array(net.params[name][0].data/net.params[name][2].data))
        l.append(np.array(net.params[name][1].data/net.params[name][2].data))

        model.get_layer(name).set_weights(l)


for layer in model.layers:
    layer.name = layer.name + "_v1"

model.save_weights("converted_model.h5")