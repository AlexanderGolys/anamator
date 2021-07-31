import math
import os

import numpy as np

from animator import basic_func, objects

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


if __name__ == '__main__':
    init()
    # basic_test()
    # test_function_sequence()
    # test_single_animator_1(save_ram=True, id='t1__', start_from=0, read_only=
    testing_dashed_line()
