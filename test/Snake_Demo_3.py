import sys
sys.path.insert(0, '../')

import BSpline_Snake.GVF as GVF
import BSpline_Snake.B_spline as B_spline
import cv2
from PIL import Image
import numpy as np
import BSpline_Snake.Snake as Snake
import matplotlib.pyplot as plot
image = cv2.imread("../test_image/test.jpg")
plot.imshow(image)
plot.title("Original Image")
plot.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plot.imshow(image,cmap='gray')
plot.title("gray_image")
plot.show()

image_size = (150,150)
GVF_generator_1 = GVF.GVF_generator(gaussian_size=3,edge_threshold=0.3,gradient_smooth=3,mu=0.5)
image_edge =  GVF_generator_1.edge_from_gray_image(gray_image,output_size=image_size)
image_GVF  =GVF_generator_1.from_gray_image(image_edge,from_edge = True)

Contour_extractor_1  = Snake.Contour_Extractor()
B_Spline_1 = Contour_extractor_1.fit_B_spline(image_GVF,image=gray_image,second_loop_break_inspect_time= None,init_control_num=100)
