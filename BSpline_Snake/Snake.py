import BSpline_Snake.B_spline as B_spline
import BSpline_Snake.GVF as GVF
import math
import time
import sys
import matplotlib.pyplot as plot
import numpy as np
class RunTimeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class Contour_Extractor():
    def __init__(self, dt = 0.05,fq_threshold =5*1e-3,f_threshold =0.1):
        self.dt = dt
        self.fq_threshold = fq_threshold
        self.f_threshold = f_threshold

    def fit_B_spline(self,image_GVF:GVF.GVF,
                     init_B_spline = None,second_loop_break_inspect_time = 10,verbose = True,image = None,verbose_interval = 1,
                     init_control_num = 4) ->B_spline.B_spline:

        if init_B_spline is None:
            w, h = image_GVF.shape
            if init_control_num <=4:
                bot,top = h*0.2 , h*0.8
                left,right =  w*0.2, w*0.8
                init_B_spline = B_spline.B_spline([[bot,left],[bot,right],[top,right],[top,left]])
            else:
                theta = np.linspace(0,360,init_control_num)[:-1]
                radius = w/2*0.8
                center = [w/2,h/2]
                init_points = np.asarray(center)  + np.asarray([[radius*math.cos(i/180*3.141),radius*math.sin(i/180*3.141)] for i in theta])
                init_B_spline = B_spline.B_spline(init_points)

        fq = init_B_spline.get_fq(image_GVF)
        verbose_start_time = time.time()
        Criterion = max([math.sqrt(i**2+j**2) for i,j in fq])
        if (verbose):
            plot.ion()
            plot.clf()
            init_B_spline.plot(image=image, fq=fq,image_resize=image_GVF.shape)
            plot.title('Criterion:%.2E, threshold:%.2E.\n' %
                       (Criterion, self.fq_threshold))
            plot.show()
            plot.pause(0.0001)

        break_start_time = time.time()
        break_min = float('inf')
        break_max = float('-inf')

        flag = True
        while flag:
            if time.time() > verbose_start_time + verbose_interval:
                verbose_start_time = time.time()
                if (verbose):
                    sys.stdout.write('Spline iteration: Criterion is:%.2E, threshold is %.2E.\n' %
                                     (Criterion, self.fq_threshold))
                    sys.stdout.flush()
                    plot.clf()
                    init_B_spline.plot(image=image,fq=fq,image_resize=image_GVF.shape)
                    plot.title('Criterion:%.2E, threshold:%.2E.\n' %
                                     (Criterion, self.fq_threshold))
                    plot.show()
                    plot.pause(0.0001)

            init_B_spline.reset_control_points(init_B_spline.control_points+fq*self.dt)
            fq = init_B_spline.get_fq(image_GVF)
            Criterion = max([math.sqrt(i ** 2 + j ** 2) for i, j in fq])
            flag = Criterion>self.fq_threshold
            if not flag:
                sys.stdout.write("Spline Fit: First Type Termination.\n")
                sys.stdout.flush()

            break_min = min(break_min,Criterion)
            break_max = max(break_max,Criterion)
            if second_loop_break_inspect_time is not None and second_loop_break_inspect_time > 0:
                if time.time()>break_start_time + second_loop_break_inspect_time:
                    break_start_time = time.time()
                    if break_min/break_max > 0.5:
                        flag = False
                        sys.stdout.write("Spline Fit: Second Type Termination.\n")
                        sys.stdout.flush()
                    break_min = float('inf')
                    break_max = float('-inf')


        plot.title('Final Contour')
        plot.ioff()
        return init_B_spline

    def get_contour(self,image_GVF:GVF.GVF,init_B_spline = None,verbose = True,image = None,verbose_interval = 1,
                    second_loop_break_inspect_time=4):
        plot.clf()
        Stop_count = 0
        spline = self.fit_B_spline(image_GVF,init_B_spline,image=image,verbose_interval = verbose_interval)
        plot.ion()

        knot_index,sample_index,force_old = spline.get_sample_force(image_GVF, largest_force = True)

        verbose_start_time = time.time()
        flag = force_old>self.f_threshold
        while flag:

            spline.add_knot(knot_index,sample_index)

            spline = self.fit_B_spline(image_GVF, spline, verbose=True,image=image,verbose_interval=verbose_interval)
            plot.ion()
            knot_index, sample_index, force_new = spline.get_sample_force(image_GVF, largest_force=True)

            if time.time() > verbose_start_time + 10:
                verbose_start_time = time.time()
                if (verbose):
                    sys.stdout.write('Add knots iteration: Criterion is:%.2E, threshold is %.2E.\n' %
                                     (force_old, self.f_threshold))
                    sys.stdout.flush()

            flag = force_new > self.f_threshold
            if not flag and verbose:
                sys.stdout.write('Add knots iteration: First Type Termination\n')
                sys.stdout.flush()

            if force_new>force_old*0.9:
                Stop_count += 1
            else:
                Stop_count = 0

            if second_loop_break_inspect_time is not None and second_loop_break_inspect_time <= Stop_count:
                flag = False
                if verbose:
                    sys.stdout.write('Add knots iteration: Second Type Termination\n')
                    sys.stdout.flush()
            force_old = force_new

        plot.title("Final Contour")
        plot.ioff()

        return spline
