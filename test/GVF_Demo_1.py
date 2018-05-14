import sys
sys.path.insert(0, '../')


import BSpline_Snake.GVF as GVF
import BSpline_Snake.B_spline as B_spline
import cv2
from PIL import Image
import numpy as np
import BSpline_Snake.Snake as Snake
import matplotlib.pyplot as plot


image = cv2.imread("../test_image/test1.jpg")
plot.imshow(image)
plot.title("Original Image")
plot.show()

gray_image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
plot.imshow(gray_image,cmap='gray')
plot.title("Gray Image")
plot.show()

imagesize = (100,100)
GVF_generator_1 = GVF.GVF_generator(gaussian_size=7,edge_threshold=0.80,gradient_smooth=3,mu=0.5)
image_edge =  GVF_generator_1.edge_from_gray_image(gray_image,output_size=imagesize)
image_GVF  =GVF_generator_1.from_gray_image(image_edge,from_edge = True,verbose_interval= 0.5,pause_time= 0.1)

