import cv2
import numpy as np
import caffe
import os



os.environ['CUDA_VISIBLE_DEVICES'] = "1"

mean_image = 127.5
scale_image = 127.5
image_height = 300
image_width = 300

image_path=  "/Path to test image/"

img = cv2.imread(image_path)
ima = img

image1 = cv2.resize(img,(image_height,image_width))
img = image1
image1 = np.array(image1,dtype=np.float32)

image1[:,:,0] = (image1[:,:,0] - mean_image)/scale_image
image1[:,:,1] = (image1[:,:,1] - mean_image)/scale_image
image1[:,:,2] = (image1[:,:,2] - mean_image)/scale_image


image1 = image1[np.newaxis,:,:,:]
image1= image1.transpose(0,3,1,2)

net.blobs['data'].data[...] = image1;


net.forward()


e = net.blobs['detection_out'].data[...]
print e.shape
e = e.reshape(e.shape[2],e.shape[3])


for i in range(e.shape[0]):

    xmin = int(ima.shape[1]*e[i][3])
    ymin = int(ima.shape[0]*e[i][4])
    xmax = int(ima.shape[1]*e[i][5])
    ymax = int(ima.shape[0]*e[i][6])
    label = int(e[i][1])
    
    print xmin,xmax,ymin ,ymax
    cv2.rectangle(ima,(xmin,ymin),(xmax,ymax),(255,0,0),2)

cv2.imshow('image',ima)
cv2.waitKey(0)