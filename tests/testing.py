import math
import os
import copy
import itertools

import numpy as np

from src.anamator import basic_func, objects
from math import pi, sin, cos, exp
from src.anamator.basic_func import *
from src.anamator.objects import *


FHD = (1920, 1080)
HD = (1280, 720)


def testing_dashed_line():
    basic_func.DEBUG = True
    # surface = AxisSurface(res=(1920, 1080), x_bounds=(-1, 1), y_bounds=(-.5, 2))
    # func = objects.Function(lambda x: x**2)
    # settings = {
    #     'sampling rate': 3,
    #     'thickness': 30,
    #     'blur': 5,
    # }
    #
    # surface.blit_parametric_object(func, settings)
    # print(surface.bitmap[:, :, 1])
    # frame = Frame(res=(1920, 1080), bg_color=(0, 0, 0, 1))
    # frame.blit_surface(surface, (0, 0))
    # frame.generate_png('test2.png')

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    func = objects.Function(lambda x: math.sin(x))
    func2 = objects.Function(lambda x: x ** 3)

    settings_function = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'white'
    }
    settings_function2 = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_point = {
        'sampling rate': 1,
        'thickness': 4,
        'blur': 2,
        'color': 'white',
        'blur kernel': 'none'
    }
    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 0,
        'color': 'black',
        'blur kernel': 'none'
    }
    frame.add_axis_surface(x_bounds=(-5.1, 5.1), y_bounds=(-2.1, 2))
    frame.blit_axes(settings_axes, x_only=False)
    # frame.blit_parametric_object(func2, settings_function2)

    # frame.blit_x_grid(settings_grid, interval=.5, length=.02)
    # frame.blit_y_grid(settings_grid, interval=.05, length=.03)
    frame.axis_surface.blit_dashed_curve(func, 10, settings=settings_function)
    frame.axis_surface.blit_dashed_curve(func2, 10, settings=settings_function)
    # for x in np.linspace(-5, 5, 10):
    #     frame.axis_surface.blit_closed_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                                settings=settings_point_interior)
    #     frame.axis_surface.blit_open_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                              settings=settings_point)
    # frame.axis_surface.blit_closed_pixel_point(coords=(1, 1), radius=4, opacity=1, settings=settings_point)

    frame.blit_axis_surface()
    frame.generate_png('test_dash.png')


def basic_test():
    basic_func.DEBUG = True
    # surface = AxisSurface(res=(1920, 1080), x_bounds=(-1, 1), y_bounds=(-.5, 2))
    # func = objects.Function(lambda x: x**2)
    # settings = {
    #     'sampling rate': 3,
    #     'thickness': 30,
    #     'blur': 5,
    # }
    #
    # surface.blit_parametric_object(func, settings)
    # print(surface.bitmap[:, :, 1])
    # frame = Frame(res=(1920, 1080), bg_color=(0, 0, 0, 1))
    # frame.blit_surface(surface, (0, 0))
    # frame.generate_png('test2.png')

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    func = objects.Function(lambda x: math.sin(x))
    func2 = objects.Function(lambda x: x ** 3)

    settings_function = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'white'
    }
    settings_function2 = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_point = {
        'sampling rate': 1,
        'thickness': 4,
        'blur': 2,
        'color': 'white',
        'blur kernel': 'none'
    }
    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 0,
        'color': 'black',
        'blur kernel': 'none'
    }
    frame.add_axis_surface(x_bounds=(-5.1, 5.1), y_bounds=(-2.1, 2))
    frame.blit_axes(settings_axes, x_only=False)
    # frame.blit_parametric_object(func2, settings_function2)

    # frame.blit_x_grid(settings_grid, interval=.5, length=.02)
    # frame.blit_y_grid(settings_grid, interval=.05, length=.03)
    frame.blit_parametric_object(func, settings_function)
    # for x in np.linspace(-5, 5, 10):
    #     frame.axis_surface.blit_closed_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                                settings=settings_point_interior)
    #     frame.axis_surface.blit_open_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                              settings=settings_point)
    # frame.axis_surface.blit_closed_pixel_point(coords=(1, 1), radius=4, opacity=1, settings=settings_point)

    frame.blit_axis_surface()
    frame.generate_png('test_grid.png')


def test_single_animator_1(save_ram=False, id='a', start_from=0, read_only=False):
    def generator(t):
        frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)

        def make_foo(x):
            return lambda h: math.sin(x+3*h)

        func = objects.Function(make_foo(t))
        settings_function = {
            'sampling rate': 3,
            'thickness': 5,
            'blur': 3,
            'color': 'white'
        }
        settings_axes = {
            'sampling rate': 3,
            'thickness': 3,
            'blur': 2,
            'color': 'white'
        }
        frame.add_axis_surface(x_bounds=(-2.1, 2.1), y_bounds=(-2.1, 2.1))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_parametric_object(func, settings_function)
        frame.blit_axis_surface()
        return frame

    def diff(t):
        return t*(3-t)

    animation = basic_func.SingleAnimation(generator, diff)
    settings = {
        'fps': 15,
        'resolution': (1280, 720),
        'duration': 3
    }
    animation.render('test_animation.mp4', settings, save_ram=save_ram, id_=id, start_from=start_from,
                     read_only=read_only)


def test_function_sequence():
    def make_poly(n):
        return lambda x: x**(2*n)

    sequence = [lambda x: x**2, lambda x: 3/2*x**2+1/4, lambda x: 2*x**2 + 1/2, lambda x: 5/2*x**2/3]
    differential = lambda x: (x-3/8)**2*(5/8-x)**2 if abs(x-1/2) < 1/8 else 0

    def generator(foo):
        frame = basic_func.OneAxisFrame((640, 480), 'black', 10, 10)
        settings_function = {
            'sampling rate': 3,
            'thickness': 10,
            'blur': 3,
            'color': 'white'
        }
        settings_axes = {
            'sampling rate': 3,
            'thickness': 5,
            'blur': 2,
            'color': 'white'
        }
        func = objects.Function(foo)
        frame.add_axis_surface(x_bounds=(-5.1, 5.1), y_bounds=(-2.1, 2))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_parametric_object(func, settings_function)
        frame.blit_axis_surface()
        return frame

    animation = basic_func.FunctionSequenceAnimation(sequence, differential, generator)
    settings = {
        'fps': 30,
        'resolution': (640, 480),
        'duration': 3
    }
    animation.render('test_seq.mp4', settings, save_ram=True, id_='seq1')


def init():
    try:
        os.mkdir('tmp')
    except FileExistsError:
        pass


def test_blitting_recs():
    blended_rec_settings = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'purple 1'
    }

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    frame.add_axis_surface((-10, 10), (-10, 10))
    rec1 = objects.FilledObject(objects.Function(function=None, const=6), objects.Function(function=None, const=-2),
                                (-4, 5))
    rec2 = objects.FilledObject(objects.Function(function=None, const=7), objects.Function(function=None, const=-8),
                                (-7, -5))
    frame.axis_surface.blit_filled_object(rec1, blended_rec_settings, queue=True)
    frame.axis_surface.blit_filled_object(rec2, blended_rec_settings, queue=True)
    frame.axis_surface.blit_filled_queue()
    frame.blit_axis_surface()
    frame.generate_png('test_blitting_recs.png')


def test_blitting_images():
    blended_rec_settings = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'purple 1'
    }

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    frame.add_axis_surface((-10, 10), (-10, 10))
    img = objects.ImageObject('img//m.png')
    frame.axis_surface.blit_bitmap_object((0, 0), img, blended_rec_settings)
    frame.axis_surface.blit_filled_queue()
    frame.blit_axis_surface()
    frame.generate_png('test_blitting_images.png')


def stupid_test():
    a = [1, 2]
    print(a[1:2])


def test_3_axis_frame():
    resolution = (1920, 1080)
    function = lambda x: (abs(x) - abs(x-1) + x**2/15)/2 + 1.2
    division = np.linspace(-4, 4, 10)
    frame = basic_func.ThreeAxisFrame(resolution, 'black', 100, 350, 200, 200)
    func = objects.Function(function)

    x_bounds = (-5, 5)
    area_bounds = (-4, 4)
    y_bounds = (-.5, 3)
    bounds = (x_bounds, y_bounds)
    frame.add_equal_axis_surfaces(bounds, bounds, bounds)

    settings_function = {
        'sampling rate': 3,
        'thickness': 6,
        'blur': 2,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }
    settings_frames = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'gray'
    }
    settings_rec_1 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'red pastel 1'
    }
    settings_rec_2 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'red pastel 2'
    }

    intervals = list(zip(division[:-1], division[1:]))
    division1 = intervals[::2]
    division2 = intervals[1:][::2]

    for settings, div in zip((settings_rec_1, settings_rec_2), (division1, division2)):
        for interval in div:
            rectangle_lower = objects.FilledObject(objects.Function(const=0),
                                                   objects.Function(const=func.inf(interval)), interval)
            frame.axis_surface_left.blit_filled_object(rectangle_lower, settings, interval, queue=True)
            rectangle_upper = objects.FilledObject(objects.Function(const=0),
                                                   objects.Function(const=func.sup(interval)), interval)
            frame.axis_surface_right.blit_filled_object(rectangle_upper, settings, interval, queue=True)

        frame.axis_surface_left.blit_filled_queue()
        frame.axis_surface_right.blit_filled_queue()

    integral = objects.FilledObject(objects.Function(const=0), func, area_bounds)
    frame.axis_surface_middle.blit_filled_object(integral, settings_rec_1, x_bounds, queue=False)

    frame.axis_surface_left.blit_parametric_object(func, settings_function)
    frame.axis_surface_right.blit_parametric_object(func, settings_function)
    frame.axis_surface_middle.blit_parametric_object(func, settings_function)

    frame.axis_surface_left.blit_axes(settings_axes, x_only=True)
    frame.axis_surface_right.blit_axes(settings_axes, x_only=True)
    frame.axis_surface_middle.blit_axes(settings_axes, x_only=True)

    frame.blit_axis_surfaces()
    frame.blit_frames_around_axis_surfaces(settings_frames, x_inner_bounds=20, y_inner_bounds=20)
    frame.generate_png('test_3_frames.png')


def stupid_test2():
    arr1 = np.array([[1, 2], [4, 5]])
    arr2 = np.array([[1, 2], [4, 5]])
    print(arr2*arr1)


def test_multi():
    def make_exp_diff(x0, c):
        return basic_func.normalize_function(lambda x: math.exp(-c*(x-x0)**2))
    # differentials_80 = [objects.Gaussian(k, 100, None) for k in np.linspace(.25, .75, 30)]
    # differentials_100 = [make_exp_diff(k, 100) for k in np.linspace(.25, .75, 30)]
    differentials_200 = [objects.Gaussian(k, 70, None) for k in np.linspace(.25, .75, 10)]*10
    differentials = differentials_200

    def radius(x):
        if x <= 3/4:
            return int(24*x)
        if x <= 1:
            return int(-32*x+40)
        else:
            return 8

    def no_radius(x):
        return int(3*x)

    settings_dots = {
        'blur': 3,
        'blur kernel': 'box'
    }

    def generate_frame(*t_list):
        frame = basic_func.OneAxisFrame((1000, 1000))
        # print(len(set([radius(t) for t in t_list])))
        bitmap_penis = \
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]
        penis = {10*y + x for x, y in itertools.product(range(10), range(10)) if bitmap_penis[9-x][y] == 1}
        dots = [objects.BitmapDisk(radius(t), 'white', 1) if i in penis else objects.BitmapDisk(no_radius(t), 'white', 1) for i, t in enumerate(t_list)]
        centers = [(int(k), int(m)) for k, m in itertools.product(np.linspace(100, 900, 10), np.linspace(100, 900, 10))]
        centers.sort(key=lambda x: 10*x[0] + x[1])
        frame.blit_distinct_bitmap_objects(centers, dots, settings_dots)
        return frame

    settings = {
        'fps': 24,
        'duration': 1,
        'resolution': (1000, 1000)
    }

    animator = basic_func.MultiDifferentialAnimation(generate_frame, *differentials)
    animator.render('test_multi2.mp4', settings, save_ram=True, speed=.5)


def test_3d(sampling_rate=1000):
    x_eq = lambda u, v: (1 + v/2*math.cos(u/2))*math.cos(u)
    y_eq = lambda u, v: (1 + v/2*math.cos(u/2))*math.sin(u)
    z_eq = lambda u, v: v/2*math.sin(u/2)
    u_bounds = (0, 2*math.pi)
    v_bounds = (-1, 1)
    mobius_strip = objects.Parametric3DObject(x_eq, y_eq, z_eq, u_bounds, v_bounds)

    def calc_g(u, v):
        w = complex(u, v)
        g1 = -1.5*(w*(1-w**4)/(w**6 + 5**.5*w**3 - 1)).imag
        g2 = -1.5*(w*(1+w**4)/(w**6 + 5**.5*w**3 - 1)).real
        g3 = ((1 + w**6)/(w**6 + 5**.5*w**3 - 1)).imag - .5
        return g1, g2, g3

    def x_eq_boys(u, v):
        if u**2 + v**2 > 1:
            return
        g1, g2, g3 = calc_g(u, v)
        return g1 / (g1**2 + g2**2 + g3**2)

    def y_eq_boys(u, v):
        if u**2 + v**2 > 1:
            return
        g1, g2, g3 = calc_g(u, v)
        return g2 / (g1**2 + g2**2 + g3**2)

    def z_eq_boys(u, v):
        if u**2 + v**2 > 1:
            return
        g1, g2, g3 = calc_g(u, v)
        return g3 / (g1**2 + g2**2 + g3**2)

    boys_surface = objects.Parametric3DObject(x_eq_boys, y_eq_boys, z_eq_boys, (-1, 1), (-1, 1))
    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    frame.add_axis_surface((-3, 3), (-2, 2))
    # frame.axis_surface.blit_3d_object(mobius_strip, objects.Parametric3DSettings(2000, 300, 'white', 'independence'))
    frame.axis_surface.blit_3d_object(boys_surface, objects.Parametric3DSettings(sampling_rate, sampling_rate, 'white', 'independence'))

    frame.blit_axis_surface()
    frame.generate_png('boys.png')
    return frame


def test_rendering_boys():
    film = basic_func.Film(1, FHD, 'boys')
    for n in range(10, 4011, 500):
        print(f'n = {n}')
        film.add_frame(test_3d(n))
    film.render('boys.mp4')


def rotate_manifold(sampling_rate, resolution):
    def frame_gen(t):
        def calc_g(u, v):
            w = complex(u, v)
            g1 = -1.5 * (w * (1 - w ** 4) / (w ** 6 + 5 ** .5 * w ** 3 - 1)).imag
            g2 = -1.5 * (w * (1 + w ** 4) / (w ** 6 + 5 ** .5 * w ** 3 - 1)).real
            g3 = ((1 + w ** 6) / (w ** 6 + 5 ** .5 * w ** 3 - 1)).imag - .5
            return g1, g2, g3

        def x_eq_boys(u, v):
            if u ** 2 + v ** 2 > 1:
                return
            g1, g2, g3 = calc_g(u, v)
            return g1 / (g1 ** 2 + g2 ** 2 + g3 ** 2)

        def y_eq_boys(u, v):
            if u ** 2 + v ** 2 > 1:
                return
            g1, g2, g3 = calc_g(u, v)
            return g2 / (g1 ** 2 + g2 ** 2 + g3 ** 2)

        def z_eq_boys(u, v):
            if u ** 2 + v ** 2 > 1.01:
                return
            g1, g2, g3 = calc_g(u, v)
            return g3 / (g1 ** 2 + g2 ** 2 + g3 ** 2)
        base = np.array([[1/3**.5, 1/3**.5, 1/3**.5],
                         [-1/2**.5, 1/2**.5, 0],
                         [1/2, 1/2, -1/2**.5]])
        inv_base = np.linalg.inv(base)
        rotation = np.array([[cos(2*pi*t), -sin(2*pi*t), 0],
                             [sin(2*pi*t), cos(2*pi*t), 0],
                             [0, 0, 1]])

        rotation_matrix = base @ rotation @ inv_base

        rotate = lambda v: v @ rotation_matrix

        boys_surface = objects.Parametric3DObject(x_eq_boys, y_eq_boys, z_eq_boys, (-1, 1), (-1, 1))
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface((-3, 3), (-2, 2))
        frame.axis_surface.blit_3d_object(boys_surface,
                                          objects.Parametric3DSettings(sampling_rate, sampling_rate, 'white',
                                                                       'independence'), point_transformation=rotate)

        frame.blit_axis_surface()
        frame.generate_png('boys.png')
        return frame

    animator = basic_func.SingleAnimation(frame_gen, objects.PredefinedSettings.slow_differential)
    animator.render('rotation.mp4', objects.RenderSettings(12, 1, resolution), save_ram=True, id_='rotate', speed=.2)


def boys_roman_homotopy(sampling_rate, resolution, id_='homotopy', filename='roman_to_boys.mp4'):
    def frame_gen(t0, t1):
        def x_eq_boys(u, v):
            return (2**.5*cos(v)**2*cos(2*u) + cos(u)*sin(2*v))/(2 - t0*2**.5*sin(3*u)*sin(2*v))

        def y_eq_boys(u, v):
            return (2**.5*cos(v)**2*sin(2*u) - sin(u)*sin(2*v))/(2 - t0*2**.5*sin(3*u)*sin(2*v))

        def z_eq_boys(u, v):
            return 3*cos(v)**2/(2 - t0*2**.5*sin(3*u)*sin(2*v))

        base = np.array([[1/3**.5, 1/3**.5, 1/3**.5],
                         [-1/2**.5, 1/2**.5, 0],
                         [1/2, 1/2, -1/2**.5]])
        inv_base = np.linalg.inv(base)
        rotation = np.array([[cos(4.1*pi*t1 + pi/8), -sin(4.1*pi*t1 + pi/8), 0],
                             [sin(4.1*pi*t1 + pi/8), cos(4.1*pi*t1 + pi/8), 0],
                             [0, 0, 1]])
        # rotation = np.array([[cos(pi/4), -sin(pi/4), 0],
        #                      [sin(pi/4), cos(pi/4), 0],
        #                      [0, 0, 1]])

        rotation_matrix = base @ rotation @ inv_base

        rotate = lambda v: v @ rotation_matrix

        boys_surface = objects.Parametric3DObject(x_eq_boys, y_eq_boys, z_eq_boys, (-pi/2, pi/2), (0, pi))
        frame = basic_func.OneAxisFrame(resolution, 'black', 0, 0)
        frame.add_axis_surface((-5, 5), (-3.5, 3.5))
        frame.axis_surface.blit_3d_object(boys_surface,
                                          objects.Parametric3DSettings(sampling_rate, sampling_rate, 'white',
                                                                       'independence'), point_transformation=rotate)

        frame.blit_axis_surface()
        frame.generate_png('boys_roman.png')
        return frame

    animator = basic_func.MultiDifferentialAnimation(frame_gen, lambda x: 1, objects.PredefinedSettings.exp_differential)
    animator.render(filename, objects.RenderSettings(12, 1, resolution), save_ram=True, id_=id_, speed=.1)


def stupid_test3():
    arr = np.zeros((2, 2, 2))
    arr[:, :, 1] = np.array([2, 2])
    print(arr)


def rescale_image_object(filename, number_of_rotations):
    img_object = objects.ImageObject(filename)
    for angle in np.linspace(0, 2*pi, number_of_rotations):
        img_object.generate_png('rotation_tests//original.png')
        img_object.rotate(angle, True).generate_png(f'rotation_tests//rotate{angle:.2f}.png')


def test_vector():
    frame = basic_func.OneAxisFrame(FHD)
    frame.add_axis_surface((-1, 1), (-1, 1))
    frame.axis_surface.blit_vector((0, 0), (-2**.5/2, 3**.5/2),
                                   objects.ParametricBlittingSettings(thickness=10, color='green 1', blur=2),
                                   objects.BitmapBlittingSettings(blur=0), position_correction=[0, 0], angle_correction=-.12)
    frame.blit_axes(objects.ParametricBlittingSettings())
    frame.blit_axis_surface()
    frame.generate_png('test_vector.png')


def test_pipeline(filename):
    pipe = lambda t: [PipeInstance(ParametricBlittingSettings(), blitting_type='axis'),
                      PipeInstance(ParametricBlittingSettings(), blitting_type='x scale', x_interval=.1, x_length=.01),
                      PipeInstance(ParametricBlittingSettings(), obj=Function(lambda x: (t+1)*x**2 + t))]
    bounds = lambda t: [[-2, 2], [-3, 3]]
    differential = [Gaussian(.5, 70, None)]
    animation = AnimationPipeline(pipe, bounds, differential)
    animation.render(filename, PipelineSettings())


def test_line_between_points():
    points = [(1, 1), (2, 3)]
    bounds = ([-5, 5], [-5, 5])
    bounds_f = lambda t: bounds
    pipe = lambda t: [PipeInstance(BitmapBlittingSettings(), [BitmapDisk(10, 'white', 1)]*2, blitting_type='bitmap', centers=points),
                      PipeInstance(ParametricBlittingSettings(), AnalyticGeometry.line_between_points(*points, interval=bounds[0]))]
    differential = [lambda x: 1]
    animation = AnimationPipeline(pipe, bounds_f, differential)
    animation.render('test_line_between_points.mp4', PipelineSettings(fps=1))

if __name__ == '__main__':
    # init()
    # basic_test()
    # test_function_sequence()
    # test_single_animator_1(save_ram=True, id='t1__', start_from=0, read_only=
    # testing_dashed_line()
    # test_blitting_recs()
    # stupid_test()
    # test_blitting_images()
    # test_3_axis_frame()
    # stupid_test2()
    # test_multi()
    # test_3d()
    # test_rendering_boys()
    # rotate_manifold(2500, HD)
    # boys_roman_homotopy(1500, HD, id_='exp_homotopy', filename='roman_to_boys_static.mp4')
    # rescale_image_object('img//turtle.png', 7)
    # test_vector()
    # test_pipeline('pipe.mp4')
    test_line_between_points()