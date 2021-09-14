import math
import numpy as np

from src.anamator.basic_func import *
from src.anamator.objects import *

HD = (1280, 720)
FHD = (1920, 1080)
SHIT = (640, 480)


def intersection(circ_ctr, pt_coords, r, d):
    mmm = math.sqrt((d**2-r**2)/(math.tan(math.pi-np.arcsin(r/d)**2+1)))
    l_trans = math.tan(math.pi-np.arcsin(r/d))*(pt_coords[0]-d-mmm)
    return pt_coords[0]-mmm, l_trans


def power_point_0(circ_center, circ_r, point_coords):

    def radius(x):
        if x <= 3/4:
            return int(14*x)
        if x <= 1:
            return int(-18*x+24)
        else:
            return 6

    def frame_generator(t):
        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))
        circle = ParametricObject(lambda x: circ_r*math.cos(x) + circ_center[0], lambda x: circ_r*math.sin(x) + circ_center[1], (0, t*2*math.pi))
        point = BitmapDisk(2*radius(t), 'white', 1)
        frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(), interval_of_param=(0, t*2*math.pi))
        frame.axis_surface.blit_bitmap_object(point_coords, point, BitmapBlittingSettings(blur=3))
        frame.blit_axis_surface()
        return frame

    animator = SingleAnimation(frame_generator, Gaussian(0.5, 250, None))
    animator.render('pp_0.mp4', RenderSettings())


def power_point_1(circ_center, circ_r, point_coords):
    d = abs(circ_center[0]-point_coords[0])
    intersec = intersection(circ_center, point_coords, circ_r, d)
    print(intersec)

    # intervals = (r_x(t),d_x(t))
    def frame_generator(interval):
        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))
        circle = ParametricObject(lambda x: circ_r * math.cos(x) + circ_center[0],
                                  lambda x: circ_r * math.sin(x) + circ_center[1])
        point = BitmapDisk(12, 'white', 1)
        frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))
        frame.axis_surface.blit_bitmap_object(point_coords, point, BitmapBlittingSettings(blur=3))
        try:
            a = -1/math.tan(math.pi-np.arcsin(circ_r/d))
            func_r = Function(lambda x: a*(x-circ_center[0])+circ_center[1])
        except ZeroDivisionError:
            func_r = Function(const=0)

        func_l = Function(const=circ_center[1])
        frame.axis_surface.blit_parametric_object(func_r, ParametricBlittingSettings(), interval_of_param=(circ_center[0], interval[0]))
        frame.axis_surface.blit_parametric_object(func_l, ParametricBlittingSettings(),
                                                  interval_of_param=(circ_center[0], interval[1]))
        frame.blit_axis_surface()
        return frame

    x_q = point_coords[0] - (circ_r/d)*(math.sqrt((d**2-circ_r**2))/math.tan(np.arcsin(circ_r/d)))

    print(x_q)

    animator = IntervalSequenceAnimation([(circ_center[0],circ_center[0]),(x_q,point_coords[0])], Gaussian(0.5, 30, None), frame_generator)
    animator.render('pp_1.mp4', RenderSettings(fps=10))


def power_point_2(circ_center, circ_r, point_coords):
    # d = abs(circ_center[0]-point_coords[0])
    d = point_coords[0]
    intersec = intersection(circ_center, point_coords, circ_r, d)
    print(intersec)
    x_q = point_coords[0] - (circ_r / d) * (math.sqrt((d ** 2 - circ_r ** 2)) / math.tan(np.arcsin(circ_r / d)))
    y_q = circ_r*math.sqrt(d**2-circ_r**2)/d
    y_q_ = math.sqrt(circ_r**2-x_q**2)
    print(x_q)
    print(y_q)
    print(y_q_)
    # print(circ_r == math.sqrt(x_q**2 + y_q**2))

    def frame_generator(t):
        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))
        circle = ParametricObject(lambda x: circ_r * math.cos(x) + circ_center[0],
                                  lambda x: circ_r * math.sin(x) + circ_center[1])
        point = BitmapDisk(12, 'white', 1)
        frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))
        frame.axis_surface.blit_bitmap_object(point_coords, point, BitmapBlittingSettings(blur=3))
        polygonal_chain = PolygonalChain([(x_q,y_q), circ_center, point_coords, (x_q, y_q)])
        frame.axis_surface.blit_parametric_object(polygonal_chain, ParametricBlittingSettings(), interval_of_param=(0, 2))
        func_power = Function(lambda x: -math.tan(np.arcsin(circ_r/d))*(x-point_coords[0]))
        frame.axis_surface.blit_parametric_object(func_power, ParametricBlittingSettings(), interval_of_param=(point_coords[0], point_coords[0]-t*(point_coords[0]-x_q)))
        # try:
        #     a = -1/math.tan(math.pi-np.arcsin(circ_r/d))
        #     func_r = Function(lambda x: a*(x-circ_center[0])+circ_center[1])
        # except ZeroDivisionError:
        #     func_r = Function(const=0)
        #
        # func_l = Function(const=circ_center[1])
        # frame.axis_surface.blit_parametric_object(func_r, ParametricBlittingSettings(), interval_of_param=(circ_center[0], interval[0]))
        # frame.axis_surface.blit_parametric_object(func_l, ParametricBlittingSettings(),
        #                                           interval_of_param=(circ_center[0], interval[1]))
        frame.blit_axis_surface()
        return frame

    # x_q = point_coords[0] - (circ_r/d)*(math.sqrt((d**2-circ_r**2))/math.tan(np.arcsin(circ_r/d)))



    # animator = IntervalSequenceAnimation([(circ_center[0],circ_center[0]),(x_q,point_coords[0])], Gaussian(0.5, 30, None), frame_generator)
    # animator.render('pp_1.mp4', RenderSettings(fps=10))
    animator = SingleAnimation(frame_generator, Gaussian(0.5, 30, None))
    animator.render('pp_2.mp4', RenderSettings(fps=10))


def malfatti_calculations(v_A, v_B, v_C):
    norm = lambda v, w: math.sqrt((v[0]-w[0])**2+(v[1]-w[1])**2)

    a = norm(v_B, v_C)
    b = norm((0,0), v_C)
    c = norm((0,0), v_B)
    s = .5 * (a + b + c)

    area = .5*c*v_C[1]
    r = area/s

    alpha = .5*np.arcsin(v_C[1]/b)
    beta = .5 * np.arcsin(v_C[1] / a)

    gamma = .5*(math.pi - 2*alpha - 2*beta)
    x_ic = c*math.tan(beta)/(math.tan(alpha)+math.tan(beta))
    incenter = (x_ic, x_ic*math.tan(alpha))

    d = norm((0, 0), incenter)
    e = norm(v_B, incenter)
    f = norm(v_C, incenter)

    r_1 = .5*r*(s-r+d-e-f)/(s-a)
    r_2 = .5*r*(s-r-d+e-f)/(s-b)
    r_3 = .5*r*(s-r-d-e+f)/(s-c)

    vector_length = lambda radius, angle, dist: radius*math.sqrt((1+1/(math.tan(angle))**2))/dist
    vector_B = [incenter[0] - v_B[0], incenter[1]]
    vector_C = [incenter[0] - v_C[0], incenter[1] - v_C[1]]
    ctr_1_pretrans = [x*vector_length(r_1, alpha, d) for x in incenter]
    ctr_2_pretrans = [v_B[0] + vector_B[0]*vector_length(r_2, beta, e), v_B[1] + vector_B[1]*vector_length(r_2, beta, e)]
    ctr_3_pretrans = [v_C[0] + vector_C[0]*vector_length(r_3, gamma, f), v_C[1] + vector_C[1]*vector_length(r_3, gamma, f)]

    ctr_1 = [ctr_1_pretrans[0] + v_A[0], ctr_1_pretrans[1] + v_A[1]]
    ctr_2 = [ctr_2_pretrans[0] + v_A[0], ctr_2_pretrans[1] + v_A[1]]
    ctr_3 = [ctr_3_pretrans[0] + v_A[0], ctr_3_pretrans[1] + v_A[1]]

    # testing image
    frame = OneAxisFrame(FHD)
    frame.add_axis_surface((-172, 172), (-88, 88))
    circle_1 = ParametricObject(lambda x: r_1 * math.cos(x) + ctr_1[0],
                              lambda x: r_1 * math.sin(x) + ctr_1[1])
    circle_2 = ParametricObject(lambda x: r_2 * math.cos(x) + ctr_2[0],
                                lambda x: r_2 * math.sin(x) + ctr_2[1])
    circle_3 = ParametricObject(lambda x: r_3 * math.cos(x) + ctr_3[0],
                                lambda x: r_3 * math.sin(x) + ctr_3[1])

    circles = [circle_1, circle_2, circle_3]
    v_B_trans = [v_B[0]+v_A[0], v_B[1]+v_A[1]]
    v_C_trans = [v_C[0]+v_A[0], v_C[1]+v_A[1]]
    # triangle = PolygonalChain([(0,0), v_B, v_C, (0,0), v_B])
    triangle = PolygonalChain([v_A, v_B_trans, v_C_trans, v_A, v_B_trans])

    # testing
    # d_line = PolygonalChain([(0,0),incenter,(0,0)])
    # e_line = PolygonalChain([v_B, incenter, v_B])
    # f_line = PolygonalChain([v_C, incenter, v_C])
    # inctr = BitmapDisk(12, 'white', 1)
    # frame.axis_surface.blit_bitmap_object(incenter, inctr, BitmapBlittingSettings(blur=3))
    # frame.axis_surface.blit_parametric_object(d_line, ParametricBlittingSettings(), interval_of_param=(0, 1))
    # frame.axis_surface.blit_parametric_object(e_line, ParametricBlittingSettings(), interval_of_param=(0, 1))
    # frame.axis_surface.blit_parametric_object(f_line, ParametricBlittingSettings(), interval_of_param=(0, 1))
    # rad_1 = PolygonalChain([ctr_1, (ctr_1[0],0), ctr_1])
    # rad_2 = PolygonalChain([ctr_2, (ctr_2[0],0), ctr_2])
    # frame.axis_surface.blit_parametric_object(rad_1, ParametricBlittingSettings(), interval_of_param=(0, 1))
    # frame.axis_surface.blit_parametric_object(rad_2, ParametricBlittingSettings(), interval_of_param=(0, 1))

    for circle in circles:
        frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))

    frame.axis_surface.blit_parametric_object(triangle, ParametricBlittingSettings(), interval_of_param=(0, 3))

    frame.blit_axis_surface()
    frame.generate_png('malfatti.png')

def malfatti_frame():
    pass


def appolonius():
    pass


if __name__ == '__main__':
    # power_point_0((0,0), 50, (160, 30))
    # power_point_2((0, 0), 50, (160, 0))
    malfatti_calculations((-120,-50),(270,0), (100,120))
    print('dupa')
