from skimage.filters import roberts, sobel, scharr
from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np



img = cv2.imread('E:\\Projects\\image_pro\\project_ip\\saray1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


roberts_img = roberts(img)
sobel_img = sobel(img)
scharr_img = scharr(img)
canny_edge = cv2.Canny(img, 103, 170)

sigma = 1
med = np.median(img)
lower = int(max(0, (1.0-sigma) * med))
upper = int(min(255, (1.0+sigma) * med))
auto_canny = cv2.Canny(img, lower, upper)
print(lower)
print(upper)


# Pre Trained model that OpenCV uses which has been trained in Caffe framework

protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load image and grab it's dimensions

image = cv2.imread('saray1.jpg')
plt.imshow(image)
(H, W) = image.shape[:2]

# consturct a blob out of the input image
# blob is basically prerocessed image
# OpenCV new dnn model contains two functions that can be used for proprecessing image and preparing them for classification via pre-trained deep learning models
# It is included scaling and mean subtraction

mean_pixel_values = np.average(image, axis=(0, 1))
blob = cv2.dnn.blobFromImage(image, scalefactor=0.4, size=(W, H),
                             mean=(
                                 mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             #mean = (),
                             swapRB=False, crop=False)

blop_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)
#cv2.imshow("Preprocessed image with blob", blop_for_plot)

net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]  # DROP other axes

hed = (255 * hed).astype("uint8")  # rescale to 0-255

plt.imshow(hed)
plt.title('HED based edge detection')
plt.show()

plt.imshow(img)
plt.title('Original Image')
plt.show()

plt.imshow(roberts_img)
plt.title('Roberts')
plt.show()

plt.imshow(sobel_img)
plt.title('Sobel')
plt.show()

plt.imshow(scharr_img)
plt.title('Scharr')
plt.show()

plt.imshow(canny_edge)
plt.title('Canny')
plt.show()

plt.imshow(auto_canny)
plt.title('Autocanny')
plt.show()

# cv2.imshow("Origins", img)
# cv2.imshow("Roberts", roberts_img)
# cv2.imshow("Sobel", sobel_img)
# cv2.imshow("Scharr", scharr_img)


# cv2.imshow('Canny', canny_edge)
# cv2.imshow('Autocanny', auto_canny)



