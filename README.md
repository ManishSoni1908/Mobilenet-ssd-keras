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


Training an SSD300 from scratch to convergence on Pascal VOC 2007 `trainval` and 2012 `trainval` produces the same mAP on Pascal VOC 2007 `test` as the original Caffe MobileNet SSD300 "07+12" model.

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
    <td align=center width="26%"><b>0.681</td>
    <td align=center width="26%"><b>0.685</td>
    <td align=center width="26%"><b>0.682</td>
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
| ![img01](./examples/000067.jpg) | ![img01](./examples/000456.jpg) |
| ![img01](./examples/001150.jpg) | ![img01](./examples/004545.jpg) |

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

    * 07+12: [SSD300*](https://drive.google.com/open?id=121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q), [SSD512*](https://drive.google.com/open?id=19NIa0baRCFYT3iRxQkOKCD7CpN6BFO8p)
    * 07++12: [SSD300*](https://drive.google.com/open?id=1M99knPZ4DpY9tI60iZqxXsAxX2bYWDvZ), [SSD512*](https://drive.google.com/open?id=18nFnqv9fG5Rh_fx6vUtOoQHOLySt4fEx)
    * COCO[1]: [SSD300*](https://drive.google.com/open?id=17G1J4zEpFwiOzgBmq886ci4P3YaIz8bY), [SSD512*](https://drive.google.com/open?id=1wGc368WyXSHZOv4iow2tri9LnB0vm9X-)
    * 07+12+COCO: [SSD300*](https://drive.google.com/open?id=1vtNI6kSnv7fkozl7WxyhGyReB6JvDM41), [SSD512*](https://drive.google.com/open?id=14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ)
    * 07++12+COCO: [SSD300*](https://drive.google.com/open?id=1fyDDUcIOSjeiP08vl1WCndcFdtboFXua), [SSD512*](https://drive.google.com/open?id=1a-64b6y6xsQr5puUsHX_wxI1orQDercM)



2. COCO models:

    * trainval35k: [SSD300*](https://drive.google.com/open?id=1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj), [SSD512*](https://drive.google.com/open?id=1IJWZKmjkcFMlvaz2gYukzFx4d6mH3py5)


3. ILSVRC models:

    * trainval1: [SSD300*](https://drive.google.com/open?id=1VWkj1oQS2RUhyJXckx3OaDYs5fx2mMCq), [SSD500](https://drive.google.com/open?id=1LcBPsd9CJbuBw4KiSuE1o1fMA-Pz2Zvw)


### ToDo

The following things are on the to-do list, ranked by priority. Contributions are welcome, but please read the [contributing guidelines](CONTRIBUTING.md).

1. Add model definitions and trained weights for SSDs based on other base networks such as MobileNet, InceptionResNetV2, or DenseNet.
2. Add support for the Theano and CNTK backends. Requires porting the custom layers and the loss function from TensorFlow to the abstract Keras backend.

Currently in the works:

* A new [Focal Loss](https://arxiv.org/abs/1708.02002) loss function.

### Important notes

* All trained models that were trained on MS COCO use the smaller anchor box scaling factors provided in all of the Jupyter notebooks. In particular, note that the '07+12+COCO' and '07++12+COCO' models use the smaller scaling factors.

### Terminology

* "Anchor boxes": The paper calls them "default boxes", in the original C++ code they are called "prior boxes" or "priors", and the Faster R-CNN paper calls them "anchor boxes". All terms mean the same thing, but I slightly prefer the name "anchor boxes" because I find it to be the most descriptive of these names. I call them "prior boxes" or "priors" in `keras_ssd300.py` and `keras_ssd512.py` to stay consistent with the original Caffe implementation, but everywhere else I use the name "anchor boxes" or "anchors".
* "Labels": For the purpose of this project, datasets consist of "images" and "labels". Everything that belongs to the annotations of a given image is the "labels" of that image: Not just object category labels, but also bounding box coordinates. "Labels" is just shorter than "annotations". I also use the terms "labels" and "targets" more or less interchangeably throughout the documentation, although "targets" means labels specifically in the context of training.
* "Predictor layer": The "predictor layers" or "predictors" are all the last convolution layers of the network, i.e. all convolution layers that do not feed into any subsequent convolution layers.
