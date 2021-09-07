import math
import os
import copy
import itertools

import numpy as np

from src.anamator import basic_func, objects


FHD = (1920, 1080)


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





if __name__ == '__main__':
    # init()
    # basic_test()
    # test_function_sequence()
    # test_single_animator_1(save_ram=True, id='t1__', start_from=0, read_only=
    # testing_dashed_line()
    # test_blitting_recs()
#     stupid_test()
#     test_blitting_images()
#     test_3_axis_frame()
#     stupid_test2()
    test_multi()