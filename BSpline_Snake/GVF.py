import numpy as np
import cv2
import scipy.ndimage as ndimage
import math
import time
import sys
import matplotlib.pyplot as plot
from collections import deque

class RunTimeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class GVF():
    def __init__(self,U,V):
        self.U = U
        self.V = V
        self.shape = U.shape

    def plot(self,image=None):
        skip = (slice(None, None, 3), slice(None, None, 3))
        size_x, size_y = self.shape
        X, Y = np.meshgrid(range(0, size_x), range(0, size_y))
        X_f = self.U[X, Y]
        Y_f = self.V[X, Y]
        plot.quiver(X[skip], Y[skip], X_f[skip], Y_f[skip])
        if image is not None:
            plot.imshow(image.transpose(), origin='low', cmap='gray')
        plot.show()

    def get_force(self, x, y):

        U_1 = self.U
        V_1 = self.V

        row_num, col_num = U_1.shape
        if x > col_num - 1:
            x = col_num - 1
        if y > row_num - 1:
            y = row_num - 1

        x = max(x, 0)
        y = max(y, 0)

        x_f = int(math.floor(x))
        x_c = int(math.ceil(x))

        y_f = int(math.floor(y))
        y_c = int(math.ceil(y))

        if x // 1 == x:
            x_w = np.asarray([1 / 2, 1 / 2])
        else:
            x_w = np.asarray([x_c - x, x - x_f])

        if y // 1 == y:
            y_w = np.asarray([1 / 2, 1 / 2])
        else:
            y_w = np.asarray([y_c - y, y - y_f])

        w = [x_w[0] * y_w[0], x_w[0] * y_w[1], x_w[1] * y_w[0], x_w[1] * y_w[1]]

        fx_v = [U_1[x_f, y_f], U_1[x_f, y_c], U_1[x_c, y_f], U_1[x_c, y_c]]
        fy_v = [V_1[x_f, y_f], V_1[x_f, y_c], V_1[x_c, y_f], V_1[x_c, y_c]]
        fx = np.asarray(w).dot(np.asarray(fx_v))
        fy = np.asarray(w).dot(np.asarray(fy_v))
        return fx, fy


class GVF_generator():

    def __init__(self,gaussian_size=3,gradient_smooth = 3,mu=0.5,dt=0.1,dx=1,dy=1,iter=1e-5):
        self.gaussian_size = gaussian_size
        self.mu = mu

        self.dx = dx
        self.dy = dy
        self.gradient_smooth = gradient_smooth

        if isinstance(dt,str) and dt.upper() =="AUTO":
            self.dt = self.dy * self.dx / (4 * self.mu * 1.2)
        else:
            self.dt = dt

        #assert self.dt <= self.dy * self.dx / (4 * self.mu), \
        #    'Hyper parameter Error:dt < dy*dx/(4*mu)'

        self.iter = iter

    def from_gray_image(self,input_image,output_size = None,verbose = True,quiver_visualization = False,
                        GVF_sparse = 5, second_loop_break_inspect_time= 10):

        ''''
        input_image -- 2d np.ndarray, or convertible
        output_size -- 
                    None: same size of input
                    tuple: size of GVF
        quiver_visualization --
                    True: plot final flow
        
        GVF_sparse --
                    Visualization parameter:
                        Density of vector flow in final visualization
                        
        second_loop_break --
                    break loop of iteration if Criterion is not increased in latest second_loop_break generation.
                    -1: close 
        '''''
        # type checking
        if  not isinstance(input_image,np.ndarray):
            input_image = np.asarray(input_image)

        if len(input_image.shape) != 2:
            raise RunTimeError("Input image should be 2 dimension!")

        #image_resize
        if output_size is not None:
            output_size = tuple(output_size)
            if len(output_size)!= 2:
                raise RunTimeError("Output_size should be a tuple (x,y)! ")

            image = cv2.resize(input_image, dsize=tuple(output_size))
        else:
            image = input_image

        #gradient_magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.gaussian_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.gaussian_size)
        gradient_magnitude = np.sqrt(sobelx * sobelx + sobely * sobely)

        #gradient of gradient_magnitude
        p = np.asarray(gradient_magnitude).astype(np.float64)
        if self.gradient_smooth>0:
            p = ndimage.filters.gaussian_filter(p, self.gradient_smooth)
        sobelx, sobely = np.gradient(p)

        sobelx = np.asarray(sobelx, dtype=np.float64)
        sobely = np.asarray(sobely, dtype=np.float64)
        gradient_magnitude2 = sobelx * sobelx + sobely * sobely

        sobelx = sobelx / math.sqrt(gradient_magnitude2.max())
        sobely = sobely / math.sqrt(gradient_magnitude2.max())
        gradient_magnitude2 = sobelx * sobelx + sobely * sobely


        #iteration
        U_1 = sobelx
        V_1 = sobely
        r = self.mu * self.dt / (self.dx * self.dy)

        flag = True
        verbose_start_time = time.time()
        break_start_time = time.time()
        break_min = float('inf')
        break_max = float('-inf')
        while (flag):
            U_Laplacian = cv2.Laplacian(U_1, cv2.CV_64F)
            U_2 = (1 - gradient_magnitude2 * self.dt) * U_1 + r * U_Laplacian + gradient_magnitude2 * sobelx * self.dt

            V_Laplacian = cv2.Laplacian(V_1, cv2.CV_64F)
            V_2 = (1 - gradient_magnitude2 * self.dt) * V_1 + r * V_Laplacian + gradient_magnitude2 * sobely * self.dt

            Criterion = ((V_2 - V_1) ** 2 + (U_2 - U_1) ** 2) / (V_1 ** 2 + V_2 ** 2 + 1e-6*self.iter)
            Criterion = math.sqrt(Criterion.max())

            break_min = min(break_min, Criterion)
            break_max = max(break_max, Criterion)

            U_1 = U_2
            V_1 = V_2

            flag = (Criterion > self.iter )
            if verbose and not flag:
                sys.stdout.write("GVF Done : First type Termination.\n")
                sys.stdout.flush()

            if time.time() > verbose_start_time + 10:
                verbose_start_time = time.time()
                if (verbose):

                    sys.stdout.write('GVF iteration:Criterion is:%.2E, threshold is %.2E.' %
                                     (Criterion,self.iter))
                    sys.stdout.write('(mu:%.2f,dt:%.2f,dx:%.2f,dy:%.2f,r:%.2f) \n' %
                                     (self.mu,self.dt,self.dx,self.dy,r))
                    sys.stdout.flush()

            if time.time() > break_start_time + second_loop_break_inspect_time:
                break_start_time = time.time()
                flag = flag and break_max/break_min > 2
                if verbose and break_max/break_min <= 2:
                    sys.stdout.write("GVF Done : Second type Termination.\n")
                    sys.stdout.flush()
                break_min = float('inf')
                break_max = float('-inf')

        if (quiver_visualization):
            skip = (slice(None, None, GVF_sparse), slice(None, None, GVF_sparse))
            size_x , size_y = U_1.shape
            X, Y = np.meshgrid(range(0, size_x), range(0, size_y))
            X_f = U_1[X, Y]
            Y_f = V_1[X, Y]
            plot.quiver(X[skip], Y[skip], X_f[skip], Y_f[skip], )
            plot.imshow(image.transpose(), origin='low', cmap='gray')
            plot.draw()

        return GVF(U_1,V_1)