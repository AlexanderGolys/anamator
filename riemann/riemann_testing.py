import math

import numpy as np

import animator.basic_func as basic_func
import animator.objects as objects


def find_sup(foo, interval, precision=50):
    return max([foo(x) for x in np.linspace(*interval, precision)])


def find_inf(foo, interval, precision=50):
    return -find_sup(lambda x: -foo(x), interval, precision)


def generate_lower_sum_frame(function, division):
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    func = objects.Function(function)

    x_bounds = (-5.3, 5.3)
    y_bounds = (-.5, 2.1)
    frame.add_axis_surface(x_bounds=x_bounds, y_bounds=y_bounds)

    settings_function = {
        'sampling rate': 3,
        'thickness': 6,
        'blur': 2,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 2,
        'color': 'white'
    }
    settings_rec_1 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'green 1'
    }
    settings_rec_2 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'green 2'
    }

    def make_const_foo(c):
        return lambda x: c

    intervals = list(zip(division[:-1], division[1:]))
    division1 = intervals[::2]
    division2 = intervals[1:][::2]
    for interval in division1:
        rectangle = objects.FilledObject(objects.Function(lambda x: 0),
                                         objects.Function(make_const_foo(find_inf(function, interval))), interval)
        frame.axis_surface.blit_filled_object(rectangle, settings_rec_1, interval, queue=True)
    frame.axis_surface.blit_filled_queue()
    for interval in division2:
        rectangle = objects.FilledObject(objects.Function(lambda x: 0),
                                         objects.Function(make_const_foo(find_inf(function, interval))), interval)
        frame.axis_surface.blit_filled_object(rectangle, settings_rec_2, interval, queue=True)
    frame.axis_surface.blit_filled_queue()

    # frame.blit_parametric_object(func2, settings_function2)
    # frame.blit_x_grid(settings_grid, interval=.5, length=.02)
    # frame.blit_y_grid(settings_grid, interval=.05, length=.03)
    frame.blit_parametric_object(func, settings_function)
    frame.blit_axes(settings_axes, x_only=True)

    first_dash = objects.ParametricObject(lambda x: -5, lambda x: x, y_bounds)
    second_dash = objects.ParametricObject(lambda x: 5, lambda x: x, y_bounds)

    frame.axis_surface.blit_dashed_curve(first_dash, 50, 50, settings_axes, y_bounds, queue=True)
    frame.axis_surface.blit_dashed_curve(second_dash, 50, 50, settings_axes, y_bounds, queue=False)

    # for x in np.linspace(-5, 5, 10):
    #     frame.axis_surface.blit_closed_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                                settings=settings_point_interior)
    #     frame.axis_surface.blit_open_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                              settings=settings_point)
    # frame.axis_surface.blit_closed_pixel_point(coords=(1, 1), radius=4, opacity=1, settings=settings_point)

    frame.blit_axis_surface()
    frame.generate_png('test_riemann_regular_dash.png')
    return frame


def make_lower_sum_film():
    film = basic_func.Film(5, (1280, 720), id='test_riemann_dash')
    # division = sorted(list(np.random.uniform(-5.1, 5.1, 5)) + [-5.1, 5.1])
    for n in range(3, 61):
        basic_func.debug(f'[{n + 1}/60]', False)
        division = [-5 + 10 * k / int(n * math.log(n)) for k in range(int(n * math.log(n)) + 1)]
        f = generate_lower_sum_frame(lambda x: math.exp(x / 10) * math.sin(x + 1) ** 2 + 0.2,
                                     division)
        # division = sorted(division + list(np.random.uniform(-5.1, 5.1, 5)))
        film.add_frame(f, save_ram=True)
    film.render('riemann_regular_dash.mp4', save_ram=True)


if __name__ == '__main__':
    make_lower_sum_film()
    # generate_lower_sum_frame(lambda x: math.exp(x/10)*math.sin(x)**2+0.2, sorted(list(np.random.uniform(-5.1, 5.1, 30)) + [-5.1, 5.1]))