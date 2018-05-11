import BSpline_Snake.GVF as GVF
import BSpline_Snake.B_spline as B_spline
import cv2
from PIL import Image
import numpy as np
import BSpline_Snake.Snake as Snake
import matplotlib.pyplot as plot
image = cv2.imread("1.png")


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = np.rot90(gray_image)
gray_image = cv2.resize(gray_image,(150,150))

image_GVF = GVF.GVF_generator(gaussian_size=3,gradient_smooth=3,mu=0.3).from_gray_image(gray_image)
image_GVF.plot()
spline = Snake.Contour_Extractor().get_contour(image_GVF,image = gray_image)
