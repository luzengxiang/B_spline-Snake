import BSpline_Snake.B_spline as B_spline
import BSpline_Snake.GVF as GVF
import math
import time
import sys
import matplotlib.pyplot as plot
import cv2
class RunTimeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class Contour_Extractor():
    def __init__(self, dt = 0.05,fq_threshold =1e-3,f_threshold =1e-3):
        self.dt = dt
        self.fq_threshold = fq_threshold
        self.f_threshold = f_threshold

    def fit_B_spline(self,image_GVF:GVF.GVF,init_B_spline = None,verbose = True,image = None) ->B_spline.B_spline:

        if init_B_spline is None:
            w,h = image_GVF.shape
            bot,top = h*0.2 , h*0.8
            left,right =  w*0.2, w*0.8
            init_B_spline = B_spline.B_spline([[bot,left],[bot,right],[top,right],[top,left]])

        fq = init_B_spline.get_fq(image_GVF)
        verbose_start_time = time.time()
        Criterion = max([math.sqrt(i**2+j**2) for i,j in fq])
        plot.ion()
        while Criterion>self.fq_threshold:
            if time.time() > verbose_start_time + 1:
                verbose_start_time = time.time()
                if (verbose):
                    sys.stdout.write('Spline iteration: Criterion is:%.2E, threshold is %.2E.\n' %
                                     (Criterion, self.fq_threshold))
                    sys.stdout.flush()
                    plot.clf()
                    init_B_spline.plot_spline(image=image,fq=fq)
                    plot.title('Criterion:%.2E, threshold:%.2E.\n' %
                                     (Criterion, self.fq_threshold))
                    plot.show()
                    plot.pause(0.001)


            init_B_spline.reset_control_points(init_B_spline.control_points+fq*self.dt)
            fq = init_B_spline.get_fq(image_GVF)
            Criterion = max([math.sqrt(i ** 2 + j ** 2) for i, j in fq])

        return init_B_spline

    def get_contour(self,image_GVF:GVF.GVF,init_B_spline = None,verbose = True,image = None):

        spline = self.fit_B_spline(image_GVF,init_B_spline,verbose = False)
        knot_index,sample_index,force = spline.get_sample_force(image_GVF, largest_force = True)

        verbose_start_time = time.time()
        while force>self.f_threshold:
            if time.time() > verbose_start_time + 10:
                verbose_start_time = time.time()
                if (verbose):
                    sys.stdout.write('Add knots iteration: Criterion is:%.2E, threshold is %.2E.\n' %
                                     (force, self.f_threshold))
                    sys.stdout.flush()
            knot_index, sample_index, force = spline.get_sample_force(image_GVF, largest_force=True)

            spline.add_knot(knot_index,sample_index)

            spline = self.fit_B_spline(image_GVF, spline, verbose=True,image=image)

        return spline
