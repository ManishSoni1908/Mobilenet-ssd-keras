import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
import cv2
import numpy as np
import keras
import os
from models.ssd_mobilenet import ssd_300




os.environ['CUDA_VISIBLE_DEVICES'] = "1"


mean_image = 127.5
scale_image = 127.5
image_height = 300
image_width = 300

orig_images = []  # Store the images here.
input_images = []  # Store resized versions of the images here.


img = cv2.imread("/home/manish/MobileNet-SSD/images/004545.jpg")
orig_images.append(img)

ima = img
image1 = cv2.resize(img,(image_height,image_width))
image1 = np.array(image1,dtype=np.float32)

image1[:,:,0] = (image1[:,:,0] - mean_image)/scale_image
image1[:,:,1] = (image1[:,:,1] - mean_image)/scale_image
image1[:,:,2] = (image1[:,:,2] - mean_image)/scale_image


image1 = image1[np.newaxis,:,:,:]
input_images = np.array(image1)


start_time = time.time()


y_pred = model.predict(input_images)

print "time taken by ssd", time.time() - start_time

confidence_threshold = 0.25

y_pred_decoded = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]


    

for box in y_pred_decoded[0]:

    xmin = int(box[-4] * orig_images[0].shape[1] / img_width)
    ymin = int(box[-3] * orig_images[0].shape[0] / img_height)
    xmax = int(box[-2] * orig_images[0].shape[1] / img_width)
    ymax = int(box[-1] * orig_images[0].shape[0] / img_height)

  
    print xmin,xmax,ymin,ymax
    cv2.rectangle(orig_images[0],(xmin, ymin), (xmax, ymax),(0,255,255),2)
  



cv2.imshow("image",orig_images[0])
cv2.waitKey(0)