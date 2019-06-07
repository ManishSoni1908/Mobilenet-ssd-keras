import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
import keras
import numpy as np 
import cv2
import keras.backend as K
import keras.layers as KL
from models.depthwise_conv2d import DepthwiseConvolution2D
from keras.models import Model
from keras.layers import Input, Lambda, Activation,Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,BatchNormalization, Add, Conv2DTranspose
from keras.regularizers import l2
from models.mobilenet_v1 import mobilenet
from misc.keras_layer_L2Normalization import L2Normalization
from misc.keras_layer_AnchorBoxes import AnchorBoxes
from misc.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def ssd_300(mode,
            image_size,
            n_classes,
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            limit_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=False,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=True,
            return_predictor_sizes=False):
    

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers



    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(lambda z: z, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(lambda z: z - np.array(subtract_mean), output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(lambda z: z / np.array(divide_by_stddev), output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels and (img_channels == 3):
        x1 = Lambda(lambda z: z[..., ::-1], output_shape=(img_height, img_width, img_channels),
                    name='input_channel_swap')(x1)


    conv4_3_norm , fc7 ,test= mobilenet(input_tensor=x1)

    print ("conv11 shape: ", conv4_3_norm.shape)
    print ("conv13 shape: ", fc7.shape)



    conv6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv14_1', use_bias=False)(fc7)
    conv6_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv14_1/bn')(conv6_1)
    conv6_1 = Activation('relu', name='relu_conv6_1')(conv6_1)

    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv14_2', use_bias=False)(conv6_1)
    conv6_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv14_2/bn')(conv6_2)
    conv6_2 = Activation('relu', name='relu_conv6_2')(conv6_2)

    print ('conv14 shape', conv6_2.shape)



    conv7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv15_1',use_bias=False)(conv6_2)
    conv7_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv15_1/bn')(conv7_1)
    conv7_1 = Activation('relu', name='relu_conv7_1')(conv7_1)

    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv15_2',use_bias=False)(conv7_1)
    conv7_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv15_2/bn')(conv7_2)
    conv7_2 = Activation('relu', name='relu_conv7_2')(conv7_2)


    print ('conv15 shape', conv7_2.shape)

    conv8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv16_1',use_bias=False)(conv7_2)
    conv8_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv16_1/bn')(conv8_1)
    conv8_1 = Activation('relu', name='relu_conv8_1')(conv8_1)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv16_2',use_bias=False)(conv8_1)
    conv8_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv16_2/bn')(conv8_2)
    conv8_2 = Activation('relu', name='relu_conv8_2')(conv8_2)

    print ('conv16 shape', conv8_2.shape)
    
    conv9_1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv17_1',use_bias=False)(conv8_2)
    conv9_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv17_1/bn')(conv9_1)
    conv9_1 = Activation('relu', name='relu_conv9_1')(conv9_1)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(conv9_1)
    conv9_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg), name='conv17_2',use_bias=False)(conv9_1)
    conv9_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv17_2/bn')(conv9_2)
    conv9_2 = Activation('relu', name='relu_conv9_2')(conv9_2)

    print ('conv17 shape', conv9_2.shape)

    # Feed conv4_3 into the L2 normalization layer
    # conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3_norm)
   

    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv11_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='conv13_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv11_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='conv13_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_loc')(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], limit_boxes=limit_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    limit_boxes=limit_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], limit_boxes=limit_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], limit_boxes=limit_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], limit_boxes=limit_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], limit_boxes=limit_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=x, outputs=predictions)
    # return model

    if mode == 'inference':
        print ('in inference mode')
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=0.01,
                                                   iou_threshold=0.45,
                                                   top_k=100,
                                                   nms_max_output_size=100,
                                                   coords='centroids',
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        print ('in training mode')

    return model







if __name__ == '__main__':
    # model = mobilenet(None)

    # for layer in model.layers:
    #     print (layer.name)


    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
    swap_channels = True  # The color channel order in the original SSD is BGR
    n_classes = 2  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                  1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
                   1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales = scales_coco
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
    coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
    normalize_coords = True

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

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
                    divide_by_stddev=None,
                    swap_channels=swap_channels)













