# MobileNet-SSD300-Keras

## SSD: A keras implementation of Mobilenet Single-Shot MultiBox Detector 

### Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Examples](#examples)
4. [Dependencies](#dependencies)
5. [Repository Content](#Repository-Content)
6. [Download the convolutionalized MobileNet-V1 weights](#download-the-convolutionalized-MobileNet-V1-weights)
7. [Download the original trained model weights](#download-the-original-trained-model-weights)
8. [Terminology](#terminology)

### Overview

This is a Keras port of the  Mobilenet SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

Weights are ported from caffe implementation of MobileNet SSD. MAP comes out to be same if we train the model from scratch and the given this implies that implementation is correct.

The repository currently provides the following network architectures:
* SSD300_mobilenet: [`ssd_mobilenet.py`](models/ssd_mobilenet.py)
 Mobilebet-V1 is used as a backbone for feature extyraction. This Network has capibility to train faster and results in increment in fps while deployment.


### Performance

Here are the mAP evaluation results of the ported weights and below that the evaluation results of a model trained from scratch using this implementation. All models were evaluated using the official Pascal VOC test server (for 2007 `test`). In all cases the results match those of the original Caffe models. Download links to all ported weights are available further below.

<table width="70%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Mean Average Precision</td>
  </tr>
  <tr>
    <td>evaluated on</td>
    <td colspan=2 align=center>VOC2007 test</td>
  </tr>
  <tr>
    <td>trained on<br>IoU rule</td>
    <td align=center width="25%">07+12<br>0.5</td>
    <td align=center width="25%">07+12+COCO<br>0.5</td>
  </tr>
  <tr>
    <td><b>MobileNet-SSD300</td>
    <td align=center><b>68.5</td>
    <td align=center><b>72.7</td>
  </tr>
</table>


Training an SSD300 from scratch to MS-COCO and then fine tune  on Pascal VOC 2007 `trainval` and 2012 `trainval` produces the same mAP on Pascal VOC 2007 `test` as the original Caffe MobileNet SSD300 "07+12+COCO" model.

<table width="95%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Mean Average Precision</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Original Caffe Model</td>
    <td align=center>Ported Weights</td>
    <td align=center>Trained from Scratch</td>
  </tr>
  <tr>
    <td><b>MobileNet-SSD300 "07+12"</td>
    <td align=center width="26%"><b>72.5</td>
    <td align=center width="26%"><b>72.7</td>
    <td align=center width="26%"><b>72.2</td>
  </tr>
</table>

The models achieve the following average number of frames per second (FPS) on Pascal VOC on an NVIDIA GeForce GTX 1080 Ti(i.e. the laptop version) and cuDNN v6 and on Nvidia Jetson Tx1.Batch Size is kept 1 for getting the prediction time which is meaningful.

<table width>
  <tr>
    <td></td>
    <td colspan=3 align=center>Frames per Second</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Nvidia 1080 Ti</td>
    <td colspan=2 align=center>Nvidia Jetson TX1</td>
  </tr>
  <tr>
    <td width="14%">Batch Size</td>
    <td width="27%" align=center>1</td>
    <td width="27%" align=center>1</td>
  </tr>
  <tr>
    <td><b>MobileNet-SSD300</td>
    <td align=center><b>170</td>
    <td align=center><b>36</td>
  </tr>
</table>

### Examples

Below are some prediction examples of the fully trained original MobileNet-SSD300 "07+12" model (i.e. trained on Pascal VOC2007 `trainval` and VOC2012 `trainval`). The predictions were made on Pascal VOC2007 `test`.

| | |
|---|---|
| ![img01](./examples/000067_result.jpg) | ![img01](./examples/000456_result.jpg) |
| ![img01](./examples/001150_result.jpg) | ![img01](./examples/004545_result.jpg) |

### Dependencies

* Python 2.x or 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV


### Repository Content

This repository provides python files that explain training, inference and evaluation.

How to use a trained model for inference:
* [`infer_mobilenet_ssd.py`](./inference/infer_mobilenet_ssd.py)

How to train a model:
* [`train_mobilenet_ssd.py`](./training/train_mobilenet_ssd.py)


How to evaluate a trained model:
* [`evaluation_mobilenet_ssd.py`](./evaluation/evaluate_mobilenet_ssd.py)

How to use the data generator:
* ['ssd_batch_generator.py'](./misc/ssd_batch_generator.py)

#### Training details

The general training setup is layed out and explained in [`train_mobilenet_ssd`](./training/train_mobilenet_ssd.py).

To train the original MobileNet-SSD300 model on Pascal VOC:

1. Download the datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Download the weights for the convolutionalized MobileNet-V1 or for one of the trained original models provided below.
3. Set the file paths for the datasets and model weights accordingly in [`train_mobilenet_ssd`](./training/train_mobilenet_ssd.py) and execute it.



### Download the convolutionalized MobileNet-V1 weights

In order to train an MobileNet-SSD300 from scratch, download the weights of the fully convolutionalized MobileNet-V1 model trained to convergence on ImageNet classification here:

[`MobileNet-v1.h5`](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

As with all other weights files below, this is a direct port of the corresponding `.caffemodel` file that is provided in the repository of the original Caffe implementation.

### Download the original trained model weights

Here are the ported weights for all the original trained models. The filenames correspond to their respective `.caffemodel` counterparts. The asterisks and footnotes refer to those in the README of the [original Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd#models).

1. PASCAL VOC models:

    * 07+12: [MobileNet-SSD300](./conversion/converted_model.h5)
    * 07++12+COCO: [MobileNet-SSD300](./conversion/converted_model.h5)


### ToDo

The following things are on the to-do list, ranked by priority. Contributions are welcome, but please read the [contributing guidelines](CONTRIBUTING.md).

1. Add model definitions and trained weights for SSDs based on other base networks such as MobileNet, InceptionResNetV2, or DenseNet.
2. Add support for the Theano and CNTK backends. Requires porting the custom layers and the loss function from TensorFlow to the abstract Keras backend.


#### Special Mention to Pierluigi Ferrari which has developed ssd keras code, it helps alot to do build this repository.
