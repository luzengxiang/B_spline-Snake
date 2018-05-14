import numpy as np
import matplotlib.pyplot as plot
import BSpline_Snake.GVF as GVF
import math
import cv2
class RunTimeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def get_sample_s(sample_size):

    sample_s = np.linspace(0, 1, sample_size+1)[:-1]

    return sample_s


def get_spline(M_rs: np.array, control_point):

    assert control_point.shape == (4,2), "the number of control point should be 4"

    return M_rs.dot(np.asarray(control_point))


class B_spline():

    Spline_Matrix = np.asarray([[-1 / 6, 0.5, -0.5, 1 / 6],
                                [0.5, -1, 0.5, 0],
                                [-0.5, 0, 0.5, 0],
                                [1 / 6, 2 / 3, 1 / 6, 0]])

    def __init__(self, control_points,sample_number: int= 10 ,sample_s=None):

        if control_points is not None:
            if not isinstance(control_points, np.ndarray):
                control_points = np.asarray(control_points)

            assert control_points.shape[1] == 2, 'Control point should be an array of length 2 array [[1,2],...]'
            assert control_points.shape[0] >=4, "Number of control points should be greater or equal than 4"
        self.control_points = control_points

        if sample_s is None:
            if sample_number is not None and sample_number>0 :
                self.sample_s = get_sample_s(sample_number)
            else:
                raise RunTimeError("Invalid sample points. Please specify either sample_num or sample_s")
        else:
            self.sample_s =sample_s

        self.M_rs =  np.asarray([[i**3,i**2,i,1]for i in self.sample_s]).dot(self.Spline_Matrix)
        self.sample_points = None
        if self.control_points is not None:
            self.get_sample_s()

    def reset_sample_s(self,sample_number: int= 10 , sample_s = None):
        if sample_s is None:
            if sample_number is not None and sample_number>0 :
                self.sample_s = get_sample_s(sample_number)
            else:
                raise RunTimeError("Invalid sample points. Please specify either sample_num or sample_s")
        else:
            self.sample_s =sample_s

        self.M_rs =  np.asarray([[i**3,i**2,i,1]for i in sample_s]).dot(self.Spline_Matrix)
        self.get_sample_s()
        return self.M_rs

    def reset_control_points(self,control_points):

        if control_points is not None:
            if not isinstance(control_points, np.ndarray):
                control_points = np.asarray(control_points)

            assert control_points.shape[1] == 2, 'Control point should be an array of length 2 array [[1,2],...]'
            assert control_points.shape[0] >=4, "Number of control points should be greater or equal than 4"
        self.control_points = control_points
        self.sample_points = None
        if self.control_points is not None:
            self.get_sample_s()


    def get_sample_s(self,constraint = None):
        assert self.control_points is not None, 'Control points are not specified.'
        tot_num = len(self.control_points)

        result = []
        knots = []
        for index in range(0, tot_num):
            if index <= tot_num - 4:
                piece_index = list(range(index, index + 4))
            else:
                piece_index = [index, (index + 1) % tot_num, (index + 2) % tot_num, (index + 3) % tot_num]
            spline = get_spline(self.M_rs, self.control_points[piece_index])
            result.extend(spline.reshape([-1,2]))
            knots.extend(spline[0].reshape([-1,2]))


        self.sample_points = np.asarray(result)
        self.knots = np.asarray(knots)
        return self.sample_points



    def plot(self, show_control_points = True, show_sample_points = True,fq=None,image = None,image_resize = None):
        if image_resize is not None and image is not None:
                output_size = tuple(image_resize)
                if len(output_size) != 2:
                    raise RunTimeError("Output_size should be a tuple (x,y)! ")

                image = cv2.resize(image, dsize=tuple(output_size))

        fig = plot.gcf()
        ax = fig.add_subplot(111)
        if show_control_points and self.control_points is not None:
            assert isinstance(self.control_points, np.ndarray)
            ax.plot([i for i, j in self.control_points],
                    [j for i, j in self.control_points], 'r+', markersize=5)

        if show_sample_points and self.sample_points is not None:
            assert isinstance(self.sample_points, np.ndarray)
            ax.plot([i for i, j in self.sample_points] + [self.sample_points[0, 0]],
                    [j for i, j in self.sample_points] + [self.sample_points[0, 1]], 'b')
            ax.plot([i for i, j in self.knots],
                    [j for i, j in self.knots], 'o', color='black')
        if fq is not None:
            for p, f in zip(self.control_points, fq):
                ax.plot([p[0], p[0] + 10 * f[0]], [p[1], p[1] + 10 * f[1]], 'g')
        if image is not None:
            ax.imshow(image.transpose())
        return fig

    def add_knot(self,knot_index,sample_index):
        i = knot_index + 1
        st = self.sample_s[sample_index]
        num = len(self.control_points)
        old_control_points = self.control_points.copy()
        for j in range(i,i+3):
            alpha_j = (st+2+i-j)/3
            old_control_points[j%num,] = (1-alpha_j)*self.control_points[(j-1)%num,] + \
                                    (alpha_j)*self.control_points[j%num,]

        add_index = (i+3) % num
        a = old_control_points[:add_index, ].reshape(-1,2)
        b = self.control_points[(add_index - 1)%num, ].reshape(-1,2)
        c = old_control_points[add_index:,].reshape(-1,2)

        new_control_points = np.concatenate((a,b,c), axis=0)


        self.control_points = new_control_points
        self.get_sample_s()
        return new_control_points

    def get_sample_force(self,image_GVF:GVF.GVF,largest_force=False):
        control_point = self.control_points
        assert isinstance(control_point,np.ndarray) , 'Control points should be np.ndarray'

        tot_num = len(control_point)
        sample_num = len(self.sample_s)

        result_matrix_x = np.zeros([sample_num, tot_num], dtype=float)
        result_matrix_y = np.zeros([sample_num, tot_num], dtype=float)

        for index in range(0, tot_num):
            spline = self.sample_points[index*sample_num:((index+1)*sample_num)]
            force = [image_GVF.get_force(x, y) for x, y in spline]
            result_matrix_x[:, index] = [x for x, y in force]
            result_matrix_y[:, index] = [y for x, y in force]

        if largest_force:
            force = result_matrix_x ** 2 + result_matrix_y ** 2
            sample_index,knot_index = np.unravel_index(np.argmax(force, axis=None), force.shape)

            return knot_index, sample_index, math.sqrt(force[sample_index,knot_index])
        else:
            return result_matrix_x, result_matrix_y

    def get_fq(self,image_GVF:GVF.GVF,plotit=False):
        sample_force_x, sample_force_y = self.get_sample_force(image_GVF)
        fq_list_x, fq_list_y = self.M_rs.transpose().dot(sample_force_x), self.M_rs.transpose().dot(sample_force_y)

        fq_x = np.zeros(len(self.control_points))
        fq_y = np.zeros(len(self.control_points))
        row, col = fq_list_x.shape

        for rowi in range(0, row):
            for coli in range(0, col):
                q_num = (coli + rowi) % len(self.control_points)
                fq_x[q_num] += fq_list_x[rowi, coli]
                fq_y[q_num] += fq_list_y[rowi, coli]

        fq = np.asarray([fq_x, fq_y]).transpose()
        if plotit:
            f, ax = plot.subplots()

            skip = (slice(None, None, 5), slice(None, None, 5))
            size_x, size_y = image_GVF.shape
            X, Y = np.meshgrid(range(0, size_x), range(0, size_y))
            X_f = image_GVF.U[X, Y]
            Y_f = image_GVF.V[X, Y]
            ax.quiver(X[skip], Y[skip], X_f[skip], Y_f[skip])

            B_spline = self.sample_points.transpose()
            control_point_1 = self.control_points
            ax.plot(list(B_spline[0]) + [B_spline[0, 0]], list(B_spline[1]) + [B_spline[1, 0]])
            ax.plot(control_point_1[:, 0], control_point_1[:, 1], 'ro', )
            for i, j in B_spline.transpose():
                force_ij = image_GVF.get_force(i, j)
                ax.plot([i, i + 10 * force_ij[0]], [j, j + 10 * force_ij[1]], 'g')

            for p, f in zip(control_point_1, fq):
                ax.plot([p[0], p[0] + 10 * f[0]], [p[1], p[1] + 10 * f[1]], 'r')
            plot.show()

        return fq
