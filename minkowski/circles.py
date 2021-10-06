import math
import numpy as np

from src.anamator.basic_func import *
from src.anamator.objects import *

HD = (1280, 720)
FHD = (1920, 1080)
F4K = (3840, 2160)
SHIT = (640, 480)


# RADIUS_FOO = PredefinedSettings.radius_func_creator(12, 8)
# CIRCLES_THIC = 3

RADIUS_FOO = PredefinedSettings.radius_func_creator(24, 16)
CIRCLES_THIC = 8
PADDING = 200


def CIRCLE_SETTINGS(color='white'):
    return ParametricBlittingSettings(3, CIRCLES_THIC, color, 3)


def intersection(circ_ctr, pt_coords, r, d):
    mmm = math.sqrt((d**2-r**2)/(math.tan(math.pi-np.arcsin(r/d)**2+1)))
    l_trans = math.tan(math.pi-np.arcsin(r/d))*(pt_coords[0]-d-mmm)
    return pt_coords[0]-mmm, l_trans


def power_point_0(circ_center, circ_r, point_coords, resolution=FHD):

    # def radius(x):
    #     if x <= 3/4:
    #         return int(14*x)
    #     if x <= 1:
    #         return int(-18*x+24)
    #     else:
    #         return 6

    def frame_generator(t):
        frame = OneAxisFrame(resolution, 'black', 200, 200)
        frame.add_axis_surface((-172, 172), (-88, 88))
        circle = ParametricObject(lambda x: circ_r*math.cos(x) + circ_center[0], lambda x: circ_r*math.sin(x) + circ_center[1], (0, t*2*math.pi))
        point = BitmapDisk(RADIUS_FOO(t), 'white', 1)
        if t >= .001:
            frame.axis_surface.blit_parametric_object(circle, CIRCLE_SETTINGS(), interval_of_param=(0, t*2*math.pi))
        frame.axis_surface.blit_bitmap_object(point_coords, point, BitmapBlittingSettings(blur=3))
        frame.blit_axis_surface()
        return frame

    animator = SingleAnimation(frame_generator, Gaussian(0.5, 70, None))

    animator.render('final//pp_0_blur.mp4', RenderSettings(resolution=resolution), speed=.25)


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


def malfatti_0(v_A, v_B, v_C):
    norm = lambda v, w: math.sqrt((v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2)

    a = norm(v_B, v_C)
    b = norm((0, 0), v_C)
    c = norm((0, 0), v_B)
    s = .5 * (a + b + c)

    area = .5 * c * v_C[1]
    r = area / s

    alpha = .5 * np.arcsin(v_C[1] / b)
    beta = .5 * np.arcsin(v_C[1] / a)
    gamma = .5 * (math.pi - 2 * alpha - 2 * beta)
    x_ic = c * math.tan(beta) / (math.tan(alpha) + math.tan(beta))
    incenter = (x_ic, x_ic * math.tan(alpha))

    d = norm((0, 0), incenter)
    e = norm(v_B, incenter)
    f = norm(v_C, incenter)

    r_1 = .5 * r * (s - r + d - e - f) / (s - a)
    r_2 = .5 * r * (s - r - d + e - f) / (s - b)
    r_3 = .5 * r * (s - r - d - e + f) / (s - c)

    vector_length = lambda radius, angle, dist: radius * math.sqrt((1 + 1 / (math.tan(angle)) ** 2)) / dist
    vector_B = [incenter[0] - v_B[0], incenter[1]]
    vector_C = [incenter[0] - v_C[0], incenter[1] - v_C[1]]
    ctr_1_pretrans = [x * vector_length(r_1, alpha, d) for x in incenter]
    ctr_2_pretrans = [v_B[0] + vector_B[0] * vector_length(r_2, beta, e),
                      v_B[1] + vector_B[1] * vector_length(r_2, beta, e)]
    ctr_3_pretrans = [v_C[0] + vector_C[0] * vector_length(r_3, gamma, f),
                      v_C[1] + vector_C[1] * vector_length(r_3, gamma, f)]

    ctr_1 = [ctr_1_pretrans[0] + v_A[0], ctr_1_pretrans[1] + v_A[1]]
    ctr_2 = [ctr_2_pretrans[0] + v_A[0], ctr_2_pretrans[1] + v_A[1]]
    ctr_3 = [ctr_3_pretrans[0] + v_A[0], ctr_3_pretrans[1] + v_A[1]]

    def frame_generator(t_triangle, *t_c):
        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))
        v_B_trans = [v_B[0] + v_A[0], v_B[1] + v_A[1]]
        v_C_trans = [v_C[0] + v_A[0], v_C[1] + v_A[1]]
        triangle = PolygonalChain([v_A, v_B_trans, v_C_trans, v_A, v_B_trans])
        frame.axis_surface.blit_parametric_object(triangle, ParametricBlittingSettings(), interval_of_param=(0, 3*t_triangle))

        circle_1 = ParametricObject(lambda x: r_1 * math.cos(x) + ctr_1[0],
                                    lambda x: r_1 * math.sin(x) + ctr_1[1])
        circle_2 = ParametricObject(lambda x: r_2 * math.cos(x) + ctr_2[0],
                                    lambda x: r_2 * math.sin(x) + ctr_2[1])
        circle_3 = ParametricObject(lambda x: r_3 * math.cos(x) + ctr_3[0],
                                    lambda x: r_3 * math.sin(x) + ctr_3[1])

        circles = [circle_1, circle_2, circle_3]

        for circle, t in zip(circles, t_c):
            frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(), interval_of_param=(0, t*2*math.pi))

        frame.blit_axis_surface()
        return frame

    animator = MultiDifferentialAnimation(frame_generator, Gaussian(0.25, 70, None), Gaussian(0.5, 70, None), Gaussian(0.5, 70, None), Gaussian(0.75, 70, None))
    animator.render('malfatti_0.mp4', RenderSettings(), speed=.5)


def malfatti_1(v_A, v_B, v_C):
    vector = lambda p1, p2: (p2[0] - p1[0], p2[1] - p1[1])
    add = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])
    sc_mult = lambda a, v: (a * v[0], a * v[1])
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

    def radius(x):
        if x <= 3/4:
            return int(14*x)
        if x <= 1:
            return int(-18*x+24)
        else:
            return 6

    def frame_generator(*t):

        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))
        circle_1 = ParametricObject(lambda x: r_1 * math.cos(x) + ctr_1[0],
                                    lambda x: r_1 * math.sin(x) + ctr_1[1])
        circle_2 = ParametricObject(lambda x: r_2 * math.cos(x) + ctr_2[0],
                                    lambda x: r_2 * math.sin(x) + ctr_2[1])
        circle_3 = ParametricObject(lambda x: r_3 * math.cos(x) + ctr_3[0],
                                    lambda x: r_3 * math.sin(x) + ctr_3[1])

        circles = [circle_1, circle_2, circle_3]

        cp4 = (norm(v_C, ctr_3_pretrans)**2 - r_3**2)**.5
        cp3 = (norm(v_B, ctr_2_pretrans)**2 - r_2**2)**.5

        triangle_tangent = [(ctr_1_pretrans[0], 0),
                            (ctr_2_pretrans[0], 0),
                            (v_B[0] - (cp3**2 - (cp3*v_C[1]/a)**2)**.5, cp3*v_C[1]/a),
                            (v_B[0] - ((cp4-a)**2 - ((cp4-a)*v_C[1]/a)**2)**.5, (a-cp4)*v_C[1]/a),
                            ((ctr_1_pretrans[0]**2 - (ctr_1_pretrans[0]*v_C[1]/b)**2)**.5, ctr_1_pretrans[0]*v_C[1]/b),
                            (((cp4-b)**2 - ((cp4-b)*v_C[1]/b)**2)**.5, (b-cp4)*v_C[1]/b)]

        tr_t_trans = [(x[0]+v_A[0], x[1]+v_A[1]) for x in triangle_tangent]

        circle_tangent = [add(ctr_1_pretrans, sc_mult(r_1/(r_1+r_2), vector(ctr_1_pretrans, ctr_2_pretrans))),
                          add(ctr_2_pretrans, sc_mult(r_2/(r_3+r_2), vector(ctr_2_pretrans, ctr_3_pretrans))),
                          add(ctr_1_pretrans, sc_mult(r_1/(r_1+r_3), vector(ctr_1_pretrans, ctr_3_pretrans)))]

        c_t_trans = [(x[0]+v_A[0], x[1]+v_A[1]) for x in circle_tangent]

        v_B_trans = [v_B[0]+v_A[0], v_B[1]+v_A[1]]
        v_C_trans = [v_C[0]+v_A[0], v_C[1]+v_A[1]]
        triangle = PolygonalChain([v_A, v_B_trans, v_C_trans, v_A, v_B_trans])
        frame.axis_surface.blit_parametric_object(triangle, ParametricBlittingSettings(), interval_of_param=(0, 3))

        for circle in circles:
            frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                      interval_of_param=(0, 2 * math.pi))

        for coords, differential in zip(tr_t_trans, t[:7]):
            point = BitmapDisk(2 * radius(differential), 'white', 1)
            frame.axis_surface.blit_bitmap_object(coords, point, BitmapBlittingSettings(blur=3))

        for coords, differential in zip(c_t_trans, t[-3:]):
            point = BitmapDisk(2 * radius(differential), 'white', 1)
            frame.axis_surface.blit_bitmap_object(coords, point, BitmapBlittingSettings(blur=3))

        frame.blit_axis_surface()
        # frame.generate_png('malf2.png')
        return frame

    differentials = [Gaussian(t, 70, None) for t in np.linspace(.2, .4, 6)] + [Gaussian(t, 70, None) for t in np.linspace(.65, .8, 3)]
    # animator = MultiDifferentialAnimation(frame_generator, *[Gaussian(0.25, 70, None)]*9)
    animator = MultiDifferentialAnimation(frame_generator, *differentials)
    animator.render('malfatti_1.mp4', RenderSettings(), speed=.5)


def appolonius_radii(ctr, r, ctrs):
    radius = lambda center, p, rad: ((p[0]-center[0])**2 + (p[1]-center[1])**2)**.5 - rad
    return [radius(ctr, c, r) for c in ctrs]


def appolonius_0(ctr, r, ctrs):
    radii = appolonius_radii(ctr, r, ctrs)
    print(radii)

    # testing
    f = OneAxisFrame(FHD)
    f.add_axis_surface((-172, 172), (-88, 88))

    def create_cos(r, c):
        return lambda x: r * math.cos(x) + c

    def create_sin(r, c):
        return lambda x: r * math.sin(x) + c

    circs = [ParametricObject(lambda x: r * math.cos(x) + ctr[0], lambda x: r * math.sin(x) + ctr[1])] + [
        ParametricObject(create_cos(r_i, ctr_i[0]), create_sin(r_i, ctr_i[1])) for
        r_i, ctr_i in zip(radii, ctrs)]

    for circle in circs:
        f.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))
    f.blit_axis_surface()
    f.generate_png('appolonius.png')

    def frame_generator(*t):
        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-172, 172), (-88, 88))

        circles = [ParametricObject(lambda x: r * math.cos(x) + ctr[0], lambda x: r * math.sin(x) + ctr[1])] + [ParametricObject(create_cos(r_i, ctr_i[0]), create_sin(r_i, ctr_i[1])) for
                   r_i, ctr_i in zip(radii, ctrs)]
        for circle, differential in zip(circles, t):
            frame.axis_surface.blit_parametric_object(circle, ParametricBlittingSettings(),
                                                      interval_of_param=(0, differential * 2 * math.pi))
        frame.blit_axis_surface()
        return frame

    animator = MultiDifferentialAnimation(frame_generator, *[Gaussian(0.25, 70, None)] * 4)
    animator.render('appolonius_0.mp4', RenderSettings())


def two_circles_power_0(r1, r2, resolution=FHD, speed=1):
    circ1 = ParametricObject(lambda x: r1*math.cos(x), lambda x: r1*math.sin(x))

    def frame_generator(t):
        x_2 = (r1 - r2)*(1-t) + t*(r1**2+r2**2)**.5
        circ2 = ParametricObject(lambda x: r2*math.cos(x) + x_2, lambda x: r2*math.sin(x))
        # print(Function(lambda x: (r1**2 - x**2)**.5 - (r2**2 - (x-x_2)**2)**.5).zeros((-172, 172), precision=10000))
        zeros = Function(lambda x: (r1**2 - x**2)**.5 - (r2**2 - (x-x_2)**2)**.5).zeros((0, r1), precision=10000)
        if zeros:
            p_x = zeros[0]
            p_y = (r1**2 - p_x**2)**.5
            p_intersec = [p_x, p_y]
            point = BitmapDisk(12, 'white', 1)

        frame = OneAxisFrame(resolution)
        bounds = ((-172, 172), (-88, 88))
        frame.add_axis_surface(*bounds)
        if t == 0:
            line_intersec1 = ParametricObject(lambda x: r1, lambda x: x)
            frame.axis_surface.blit_parametric_object(line_intersec1, ParametricBlittingSettings(), interval_of_param=(-88, 88))
            t_interval = (-pi/2, pi/2)
        else:
            a1 = -p_x / (r1 ** 2 - p_x ** 2) ** .5
            b1 = p_x**2 / (r1 ** 2 - p_x ** 2) ** .5 + p_y
            a2 = (- p_x + x_2) / (r1 ** 2 - p_x ** 2) ** .5
            b2 = -p_x*(- p_x + x_2) / (r1 ** 2 - p_x ** 2) ** .5 + p_y
            # line_intersec1 = Function(lambda x: a1*x + b1)
            # line_intersec2 = Function(lambda x: a2*x + b2)
            line_intersec1 = LinearFunction(a1, b1, bounds)
            line_intersec2 = LinearFunction(a2, b2, bounds)

            frame.axis_surface.blit_parametric_object(line_intersec1, ParametricBlittingSettings())
            frame.axis_surface.blit_parametric_object(line_intersec2, ParametricBlittingSettings())
            t2 = -np.arctan(-p_x / (r1 ** 2 - p_x ** 2) ** .5)
            t_interval = (np.arctan((-p_x + x_2) / (r1 ** 2 -p_x ** 2)**.5), np.arctan(-p_x / (r1 ** 2 - p_x ** 2) ** .5) + pi)

        angle = ParametricObject(lambda x: 10*math.cos(x) + p_x, lambda x: 10*math.sin(x) + p_y)
        # the first point is from line_intersec2 (tangent to small circles), the second one from line_intersec1 (tangent to big circle)
        frame.axis_surface.blit_parametric_object(angle,
                                                      ParametricBlittingSettings(),
                                                      interval_of_param=t_interval)
        # frame.axis_surface.blit_parametric_object(angle,
        #                                       ParametricBlittingSettings(),
        #                                       interval_of_param=(
        #                                       np.arctan((-p_x + x_2) / (r1 ** 2 - p_x ** 2) ** .5),
        #                                       math.pi/2 - np.arctan(-p_x / (r1 ** 2 - p_x ** 2) ** .5)))

        # print('from r2_', np.arctan((-p_x + x_2) / (r1 ** 2 - p_x ** 2) ** .5))
        # print('from r1_', np.arctan(-p_x / (r1 ** 2 - p_x ** 2) ** .5))

        frame.axis_surface.blit_parametric_object(circ1, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))
        frame.axis_surface.blit_parametric_object(circ2, ParametricBlittingSettings(),
                                                  interval_of_param=(0, 2 * math.pi))
        if p_intersec:
            frame.axis_surface.blit_bitmap_object(p_intersec, point, BitmapBlittingSettings(blur=3))
        frame.blit_axis_surface()
        return frame

    animator = SingleAnimation(frame_generator, Gaussian(0.5, 70, None))
    animator.render('two_circ_power_0.mp4', RenderSettings(resolution=resolution), speed=speed)


if __name__ == '__main__':
    power_point_0((0,0), 50, (160, 0), F4K)
    # power_point_2((0, 0), 50, (160, 0))
    # malfatti_calculations((-120,-50),(270,0), (100,120))
    # malfatti_0((-120, -50), (270, 0), (100, 120))
    # malfatti_1((-120, -50), (270, 0), (100, 120))
    # appolonius_0((0,0), 20, [(16,-32), (38,16), (-24, 32)])
    # print(appolonius_radii((0, 0), 50, [(70, 30), (-70, -100), (-50, 200)]))
    # two_circles_power_0(50, 30, speed=.1)
    # print('dupa')
