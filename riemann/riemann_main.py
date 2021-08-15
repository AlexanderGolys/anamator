import math
import functools
import sys

import numpy as np

import src.anamator.basic_func as basic_func
import src.anamator.objects as objects

HD = (1280, 720)
FHD = (1920, 1080)


def find_sup(foo, interval, precision=50):
    return max([foo(x) for x in np.linspace(*interval, precision)])


def find_inf(foo, interval, precision=50):
    return -find_sup(lambda x: -foo(x), interval, precision)


def generate_lower_sum_frame(function, division, resolution):
    frame = basic_func.OneAxisFrame(resolution, 'black', 50, 50)
    func = objects.Function(function)

    x_bounds = (-5, 5)
    y_bounds = (-1, 2.5)
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
        'blur': 0,
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

    # first_dash = objects.ParametricObject(lambda x: x, math.sin)
    # second_dash = objects.ParametricObject(lambda x: 5, lambda x: x, y_bounds)
    #
    # frame.axis_surface.blit_dashed_curve(first_dash, number_of_dashes=90, precision=100, settings=settings_axes,
    #                                      interval_of_param=x_bounds, queue=False)
    # frame.axis_surface.blit_dashed_curve(second_dash, 50, 50, settings_axes, y_bounds, queue=False)

    # for x in np.linspace(-5, 5, 10):
    #     frame.axis_surface.blit_closed_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                                settings=settings_point_interior)
    #     frame.axis_surface.blit_open_pixel_point(coords=func.get_point(x), radius=10, opacity=1,
    #                                              settings=settings_point)
    # frame.axis_surface.blit_closed_pixel_point(coords=(1, 1), radius=4, opacity=1, settings=settings_point)

    frame.blit_axis_surface()
    frame.generate_png('test_riemann_regular_nlogn_square.png')
    return frame


def make_lower_sum_film(resolution):
    film = basic_func.Film(5, resolution, id='test_riemann_regular')
    # division = sorted(list(np.random.uniform(-5.1, 5.1, 5)) + [-5.1, 5.1])
    for n in range(3, 70):
        basic_func.debug(f'[{n+1}/70]', False)
        division = [-5+10*k/int(n*math.log(n)) for k in range(int(n*math.log(n))+1)]
        f = generate_lower_sum_frame(lambda x: math.exp(x / 10) * math.sin(x+1) ** 2 + 0.2,
                                     division, resolution)
        # division = sorted(division + list(np.random.uniform(-5.1, 5.1, 5)))
        film.add_frame(f, save_ram=True)
    film.render('riemann_regular_nlogn_square.mp4', save_ram=True)


def smooth_animate(division_seq, foo):
    def frame_gen(t):
        return generate_lower_sum_frame(foo, list((1-t+math.floor(t))*np.array(division_seq[math.floor(t)])
                                        + (t-math.floor(t))*np.array(division_seq[min(math.floor(t)+1,
                                                                                      len(division_seq)-1)])))
    differential = lambda x: (x-3/8)**2*(5/8-x)**2 if abs(x-1/2) < 1/8 else 0

    animation = basic_func.SingleAnimation(frame_gen,
                                          basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 30,
        'resolution': (640, 480),
        'duration': 5
    }
    animation.render('riemann_seq.mp4', settings, save_ram=True, id_='riem1')


def generate_integral(function, interval, x_bounds, y_bounds, function_settings, int_settings, axes_settings, res):
    def make_const(c):
        return lambda x: c

    frame = basic_func.OneAxisFrame(res, 'black', 100, 100)
    frame.add_axis_surface(x_bounds, y_bounds)
    area = objects.FilledObject(objects.Function(lambda x: 0), objects.Function(function), interval)
    frame.axis_surface.blit_filled_object(area, int_settings)
    function_obj = objects.Function(function)
    frame.blit_parametric_object(function_obj, function_settings)
    first_dash = objects.ParametricObject(make_const(interval[0]), lambda t: t)
    second_dash = objects.ParametricObject(make_const(interval[1]), lambda t: t)
    frame.axis_surface.blit_dashed_curve(first_dash, 40, 50, axes_settings, y_bounds, queue=True)
    frame.axis_surface.blit_dashed_curve(second_dash, 40, 50, axes_settings, y_bounds, queue=False)
    frame.blit_axes(axes_settings)
    frame.blit_axis_surface()
    return frame


def smooth_animate_changing_integral(function_seq, int_seq, res):
    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }
    settings_rec_2 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'green 2'
    }
    settings_function = {
        'sampling rate': 3,
        'thickness': 6,
        'blur': 2,
        'color': 'white'
    }
    x_bounds = (-5, 5)
    y_bounds = (-2, 2)

    def frame_gen(t):
        return generate_integral(lambda x: (1 - t + math.floor(t)) * function_seq[math.floor(t)](x) +
                                           (t-math.floor(t)) * function_seq[min(math.floor(t)+1, len(function_seq)-1)](x),
                                 list((1 - t + math.floor(t)) * np.array(int_seq[math.floor(t)])
                                 + (t - math.floor(t)) * np.array(int_seq[min(math.floor(t) + 1, len(int_seq) - 1)])),
                                 x_bounds, y_bounds, settings_function, settings_rec_2, settings_axes, res)

    differential = lambda x: (x - 3 / 8) ** 2 * (5 / 8 - x) ** 2 if abs(x - 1 / 2) < 1 / 8 else 0

    animation = basic_func.SingleAnimation(frame_gen,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': res,
        'duration': len(int_seq) - 1
    }

    animation.render('smooth_int.mp4', settings, save_ram=True, id_='riem1', speed=.25)


def make_const(c):
    return lambda x: c


def smooth_animate_divisions(divisions, resolution, speed):
    x_length = .02

    def generate_division_frame(division, grid_settings, axis_settings, bounds_settings):
        x_bounds = (division[0] - 1, division[-1] + 1)
        y_bounds = (-2, 2)
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        lines = [objects.ParametricObject(make_const(point), lambda t: t, [-x_length, x_length])
                 for point in division]
        grid = functools.reduce(lambda x, y: x.stack_parametric_objects(y), lines)
        frame.axis_surface.blit_parametric_object(grid, grid_settings, interval_of_param=(-x_length, (2 * len(lines) - 1) * x_length))
        first_dash = objects.ParametricObject(make_const(divisions[0][0]), lambda t: t)
        second_dash = objects.ParametricObject(make_const(divisions[0][-1]), lambda t: t)
        frame.blit_axes(axis_settings, x_only=True)
        frame.axis_surface.blit_dashed_curve(first_dash, 40, 50, bounds_settings, y_bounds, queue=True)
        frame.axis_surface.blit_dashed_curve(second_dash, 40, 50, bounds_settings, y_bounds, queue=False)
        frame.blit_axis_surface()
        return frame

    def generate_frame(t):
        settings_axes = {
            'sampling rate': 3,
            'thickness': 3,
            'blur': 2,
            'color': 'white'
        }
        settings_bounds = {
            'sampling rate': 3,
            'thickness': 2,
            'blur': 0,
            'color': 'gray'
        }
        return generate_division_frame(basic_func.SingleAnimation.blend_lists(divisions, t), settings_axes,
                                       settings_axes, settings_bounds)

    differential = lambda x: (x - 3 / 8) ** 2 * (5 / 8 - x) ** 2 if abs(x - 1 / 2) < 1 / 8 else 0
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': len(divisions) - 1
    }
    animation.render('divisions.mp4', settings, save_ram=True, id_='div', speed=speed)


def smooth_lower_sums_op(divisions, foos, resolution, speed, filename='op_lower_sums.mp4'):
    foo_settings = objects.PredefinedSettings.t2b0white
    axis_settings = objects.PredefinedSettings.t2b0white
    blended_rec_settings = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'purple 1'
    }
    blended_rec_settings_red = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'red'
    }
    dash_settings = {
        'sampling rate': 10,
        'thickness': 2,
        'blur': 2,
        'color': 'gray'
    }
    divs_and_foo = []
    for divs, foo in zip(divisions, foos):
        divs_and_foo += [(div, foo) for div in divs]
    divs = [x[0] for x in divs_and_foo]
    foos = [x[1] for x in divs_and_foo]

    def generate_frame(t):
        division = basic_func.SingleAnimation.blend_lists(divs, t)
        func = basic_func.SingleAnimation.blend_functions(foos, t)
        x_bounds = (division[0] - 1, division[-1] + 1)
        y_bounds = (-1, 3)
        frame = basic_func.OneAxisFrame(resolution, 'black', 5, 5)
        frame.add_axis_surface(x_bounds, y_bounds)

        for interval in zip(division[:-1], division[1:]):
            inf = objects.Function(make_const(find_inf(func, interval)))
            inf.add_bounds(interval)
            blended_rec = objects.FilledObject(inf, objects.Function(lambda x: 0), interval)
            frame.axis_surface.blit_filled_object(blended_rec, blended_rec_settings, interval, queue=True)
        frame.axis_surface.blit_filled_queue()

        frame.axis_surface.blit_parametric_queue()
        frame.blit_axes(axis_settings, x_only=True)
        func_obj = objects.Function(func)
        frame.blit_parametric_object(func_obj, foo_settings)
        for point in division:
            line = objects.ParametricObject(make_const(point), lambda x: x, y_bounds)
            frame.axis_surface.blit_dashed_curve(line, 50, 200, dash_settings, y_bounds, True)
        frame.blit_axis_surface()
        return frame

    differential = lambda x: (x - 3 / 8) ** 2 * (5 / 8 - x) ** 2 if abs(x - 1 / 2) < 1 / 8 else 0
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': len(divs) - 1
    }
    animation.render(filename, settings, save_ram=True, id_='op', speed=speed)


def smooth_animate_change_to_recs(division, foo, resolution, speed, filename='change_to_rec.mp4'):
    foo_settings = objects.PredefinedSettings.t10b4white
    axis_settings = objects.PredefinedSettings.t5b2white
    blended_settings = {
            'sampling rate': 3,
            'thickness': 10,
            'blur': 4,
            'color': 'purple 1'
        }
    blended_rec_settings = {
            'sampling rate': 3,
            'thickness': 0,
            'blur': 0,
            'color': 'purple 1'
        }
    dash_settings = {
            'sampling rate': 10,
            'thickness': 2,
            'blur': 2,
            'color': 'gray'
        }

    def generate_frame(t):
        x_bounds = (division[0] - 1, division[-1] + 1)
        y_bounds = (-.5, 2)
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        for interval in zip(division[:-1], division[1:]):
            # blended_foo_sup = objects.Function(basic_func.SingleAnimation.blend_functions([foo, make_const(find_sup(foo, interval))], t))
            # blended_foo_sup.add_bounds(interval)
            blended_foo_inf = objects.Function(basic_func.SingleAnimation.blend_functions([foo, make_const(find_inf(foo, interval))], t))
            blended_foo_inf.add_bounds(interval)
            blended_rec = objects.FilledObject(blended_foo_inf, objects.Function(lambda x: 0), interval)
            # frame.axis_surface.blit_parametric_object(blended_foo_sup, blended_settings, interval, queue=True)
            frame.axis_surface.blit_parametric_object(blended_foo_inf, blended_settings, interval, queue=True)
            frame.axis_surface.blit_filled_object(blended_rec, blended_rec_settings, interval, queue=True)
        frame.axis_surface.blit_parametric_queue()
        frame.axis_surface.blit_filled_queue()
        foo_obj = objects.Function(foo)
        frame.blit_parametric_object(foo_obj, foo_settings)
        frame.blit_axes(axis_settings)
        for point in division:
            line = objects.ParametricObject(make_const(point), lambda x: x, y_bounds)
            frame.axis_surface.blit_dashed_curve(line, 50, 200, dash_settings, y_bounds, True)
        frame.blit_axis_surface()
        return frame

    differential = lambda x: (x - 3 / 8) ** 2 * (5 / 8 - x) ** 2 if abs(x - 1 / 2) < 1 / 8 else 0
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 1
    }
    animation.render(filename, settings, save_ram=True, id_='div', speed=speed)


def decompose_shape(resolution, speed, filename='decompose.mp4'):
    cube_root = lambda x: x**(1/3) if x >= 0 else -(-x)**(1/3)
    foo1_ = lambda x: (x*(5-x)*math.sin(cube_root(x)) + math.exp(x/5))**.5 - 7*math.exp(-(x+15)**2/30)
    foo2_ = lambda x: -10*(1-((x-5.556)/18.28 + 1)**2)**.5 + 4*math.exp(-(x+15)**2/20)

    foo1 = lambda x: foo1_(x) if not (math.isnan(foo1_(x)) or np.iscomplex(foo1_(x))) else 0
    foo2 = lambda x: foo2_(x) if not (math.isnan(foo2_(x)) or np.iscomplex(foo2_(x))) else 0

    after1_foo1 = lambda t: lambda x: foo1(x+10*t) + 10*t
    after2_foo1 = lambda t: lambda x: foo1(x) + 10*t
    after3_foo1 = lambda t: lambda x: foo1(x-10*t) + 10*t
    after1_foo2 = lambda t: lambda x: foo2(x + 10*t) - 10*t
    after2_foo2 = lambda t: lambda x: foo2(x) - 10*t
    after3_foo2 = lambda t: lambda x: foo2(x - 10*t) - 10*t

    interval1 = lambda t: [-31 - 10*t, -18 - 10*t]
    interval2 = [-18, -6]
    interval3 = lambda t: [-6 + 10*t, 5.55 + 10*t]

    axis_settings = objects.PredefinedSettings.t2b0white
    blended_rec_settings = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'purple 1'
    }

    def generate_frame(t):
        x_bounds = (-45, 20)
        y_bounds = (-25, 30)
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)

        s1 = objects.FilledObject(objects.Function(after1_foo1(t)), objects.Function(make_const(10*t)), interval1(t))
        s2 = objects.FilledObject(objects.Function(after2_foo1(t)), objects.Function(make_const(10*t)), interval2)
        s3 = objects.FilledObject(objects.Function(after3_foo1(t)), objects.Function(make_const(10*t)), interval3(t))
        s4 = objects.FilledObject(objects.Function(after1_foo2(t)), objects.Function(make_const(-10*t)), interval1(t))
        s5 = objects.FilledObject(objects.Function(after2_foo2(t)), objects.Function(make_const(-10*t)), interval2)
        s6 = objects.FilledObject(objects.Function(after3_foo2(t)), objects.Function(make_const(-10*t)), interval3(t))

        frame.axis_surface.blit_filled_object(s1, blended_rec_settings, queue=True)
        frame.axis_surface.blit_filled_object(s2, blended_rec_settings, queue=True)
        frame.axis_surface.blit_filled_object(s3, blended_rec_settings, queue=True)
        frame.axis_surface.blit_filled_object(s4, blended_rec_settings, queue=True)
        frame.axis_surface.blit_filled_object(s5, blended_rec_settings, queue=True)
        frame.axis_surface.blit_filled_object(s6, blended_rec_settings, queue=True)

        frame.blit_axes(axis_settings, x_only=False)
        frame.blit_axis_surface()
        return frame

    differential = objects.PredefinedSettings.slow_differential
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 1
    }
    animation.render(filename, settings, save_ram=True, id_='op', speed=speed)


def get_triple_frame(division, resolution=HD):
    function = lambda x: (abs(x) - abs(x - 1) + x ** 2 / 15) / 2 + 1.2
    # division = np.linspace(-4, 4, 10)
    if resolution == FHD:
        frame = basic_func.ThreeAxisFrame(resolution, 'black', 100, 350, 200, 200)
    else:
        frame = basic_func.ThreeAxisFrame(resolution, 'black', 44, 155, 88, 88)
    func = objects.Function(function)

    x_bounds = (-5, 5)
    area_bounds = (-4, 4)
    y_bounds = (-.5, 3)
    bounds = (x_bounds, y_bounds)
    frame.add_equal_axis_surfaces(bounds, bounds, bounds)

    settings_function = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 1,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 0,
        'color': 'white'
    }
    # settings_frames = {
    #     'sampling rate': 3,
    #     'thickness': 3,
    #     'blur': 0,
    #     'color': 'gray'
    # }

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
    # frame.blit_frames_around_axis_surfaces(settings_frames, x_inner_bounds=20, y_inner_bounds=20)
    return frame


def triple_densing(densing_function, no_frames, fps, resolution=FHD):
    area_bounds = (-4, 4)
    film = basic_func.Film(fps, resolution)
    for i in range(no_frames):
        division = np.linspace(*area_bounds, densing_function(i))
        film.add_frame(get_triple_frame(division, resolution), save_ram=True)
        print(f'{i+1}/{no_frames} ({int(100*(i+1)/no_frames)}%)')
    film.render('triple_densing.mp4', save_ram=True)


def triple_moving(divisions, speed, resolution=HD, filename='triple_moving.mp4'):
    animator = basic_func.IntervalSequenceAnimation(divisions, basic_func.normalize_function(basic_func.make_periodic(objects.PredefinedSettings.slow_differential)),
                                                    lambda d: get_triple_frame(d, resolution))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': len(divisions) - 1
    }
    animator.render(filename, settings, save_ram=True, id_='op', speed=speed)


def absolute_value(func, speed, resolution, filename='abs.mp4'):
    x_bounds = (-5, 5)
    area_bounds = (-5, 5)
    function = objects.Function(func)
    abs_function = objects.Function(lambda x: abs(func(x)))
    y_bounds = (max(function.sup(x_bounds), abs_function.sup(x_bounds)) + 1,
                min(function.inf(x_bounds), 0) - 1)
    settings_rec_positive = {
        'sampling rate': 2,
        'thickness': 0,
        'blur': 0,
        'color': 'tea green'
    }
    settings_rec_negative = {
        'sampling rate': 2,
        'thickness': 0,
        'blur': 0,
        'color': 'light red'
    }

    def gen_frame(foo):
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        blended_function = objects.Function(foo)
        one_sign_intervals = blended_function.one_sign_intervals(area_bounds)
        for interval in one_sign_intervals:
            fill = objects.FilledObject(objects.Function(const=0), objects.Function(blended_function), interval)
            if blended_function(np.mean(np.asarray(interval))) >= 0:
                continue
            frame.axis_surface.blit_filled_object(fill, settings_rec_positive, queue=True)
        frame.axis_surface.blit_filled_queue()
        for interval in one_sign_intervals:
            fill = objects.FilledObject(objects.Function(const=0), objects.Function(blended_function), interval)
            if blended_function(np.mean(np.asarray(interval))) < 0:
                continue
            frame.axis_surface.blit_filled_object(fill, settings_rec_negative, queue=True)
        frame.axis_surface.blit_filled_queue()
        frame.axis_surface.blit_axes(objects.PredefinedSettings.fhd_axis)
        frame.axis_surface.blit_parametric_object(blended_function, objects.PredefinedSettings.fhd_foo)
        frame.blit_axis_surface()
        return frame

    animator = basic_func.FunctionSequenceAnimation((func, lambda t: abs(func(t))),
                                                    objects.PredefinedSettings.slow_differential, gen_frame)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 1
    }
    animator.render(filename, settings, True, 'asd', speed=speed)


def intervals_into_divisions(division, speed, resolution, filename='intervals_into_points.mp4'):
    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 0,
        'color': 'black',
        'blur kernel': 'none'
    }
    radius = lambda x: int(25*x**2 + 5/4*x) if x <= 3/4 else int(-25*x**2 + 23.75*x + 11.25)

    def frame_gen(t):
        if t <= 1:
            frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
            frame.add_axis_surface((0, 1), (0, 1))
            # for x in division:
            points = [objects.BitmapDisk(radius(t), 'white', 1)]*len(division)
            frame.axis_surface.blit_distinct_bitmap_objects(list(map(lambda x: (x, 0), division)), points,
                                                            settings_point_interior)
        else:
            frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
            frame.add_axis_surface((0, 1), (0, 1))
            # for x in division:
            points = [objects.BitmapDisk(10, 'white', 1)] * len(division)
            frame.axis_surface.blit_distinct_bitmap_objects(list(map(lambda x: (x, 0), division)), points,
                                                            settings_point_interior)

            zero = objects.Function(const=0)
            for interval in zip(division[:-1], division[1:]):
                mean = np.mean(np.asarray(interval))
                a, b = interval
                t_ = t - 1
                if not math.isclose(t_, 0):
                    frame.axis_surface.blit_parametric_object(zero, objects.PredefinedSettings.fhd_axis,
                                                              (t*a + (1-t)*mean, t*b + (1-t)*mean))
            frame.axis_surface.blit_parametric_queue()
        frame.blit_axis_surface()
        return frame

    animator = basic_func.SingleAnimation(frame_gen, objects.PredefinedSettings.slow_differential)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 2
    }
    animator.render(filename, settings, True, 'dupa', speed=speed)


if __name__ == '__main__':
    sys.setrecursionlimit(3000)
    # make_lower_sum_film()
    # seq = [(-5, -5, -5, -5, -3, -1.5, 0, 2, 3, 4, 5),
    #        (-5, -5, -5, -3, -2.5, 0.5, 1.1, 2.3, 3.2, 4.1, 5),
    #        (-5, -5, -4, -2.8, -2.5, -2, -1, 1.4, 2.2, 3.5, 5),
    #        (-5, -4, -3.5, -2.1, -.5, 0, 1.2, 2.5, 3.1, 4, 5)]
    # smooth_animate(seq, lambda x: math.exp(x/10)*math.sin(x)**2+0.2)
    # generate_lower_sum_frame(lambda x: math.exp(x/10)*math.sin(x)**2+0.2, sorted(list(np.random.uniform(-5.1, 5.1, 30)) + [-5.1, 5.1]))
    # func_seq = [math.sin, lambda x: math.exp(x/10)*math.sin(x)**2+0.2, math.cos, lambda x: math.exp(x/10)]
    # int_seq = [(-4, 3), (-4.5, 2), (-3, 4), (-1, 4.5)]
    # smooth_animate_changing_integral(func_seq, int_seq, (1280, 720))
    # divisions = [(0, 1, 1, 3, 3, 3, 4),
    #              (0, 1.5, 2, 2.5, 3.5, 3.7, 4),
    #              (0, 1, 1, 3, 3.4, 3.4, 4),
    #              (0, 2, 2, 2, 3, 3.2, 4)]
    # smooth_animate_divisions(divisions, (1280, 720), 1)
    # division_ = [-4, -3.5, -3, -2, -1.7, -1.3, -1, -.3, .5, 1.5, 2, 2.2, 2.4, 2.6, 2.9, 3, 3.8, 4]
    # smooth_animate_change_to_recs(division_, lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 0.2, (1280, 720), .15,
    #                               'change_to_rec.mp4')
    # for a, b in zip(division_[:-1], division_[1:]):
    #     division_.append((a + b) / 2)
    # division_.sort()
    # smooth_animate_change_to_recs(division_, lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 0.2, (1280, 720), .15,
    #                               'change_to_rec_dense.mp4')
    #
    # foos = [lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 1.2, lambda x: x*(x-2)*(x+2)*(x-5)*(x+5)*math.exp(-abs(x))*math.sin(x)/12+1,
    #         lambda x: (abs(x) - abs(x-1) + x**2/15)/2 + 1.2]
    # divs = [[(-5, -5, -5, -5, -3, -1.5, 0, 2, 3, 4, 5),
    #         (-5, -5, -5, -3, -2.5, 0.5, 1.1, 2.3, 3.2, 4.1, 5),
    #         (-5, -5, -4, -2.8, -2.5, -2, -1, 1.4, 2.2, 3.5, 5),
    #         (-5, -4, -3.5, -2.1, -.5, 0, 1.2, 2.5, 3.1, 4, 5)],
    #         [(-5, -5, -5, -5, -3, -1.5, 0, 2, 3, 4, 5),
    #          (-5, -5, -5, -3, -2.5, 0.5, 1.1, 2.3, 3.2, 4.1, 5),
    #          (-5, -5, -4, -2.8, -2.5, -2, -1, 1.4, 2.2, 3.5, 5),
    #          (-5, -4, -3.5, -2.1, -.5, 0, 1.2, 2.5, 3.1, 4, 5)][::-1],
    #         [(-5, -5, -5, -5, -3, -1.5, 0, 2, 3, 4, 5),
    #          (-5, -5, -5, -3, -2.5, 0.5, 1.1, 2.3, 3.2, 4.1, 5),
    #          (-5, -5, -4, -2.8, -2.5, -2, -1, 1.4, 2.2, 3.5, 5),
    #          (-5, -4, -3.5, -2.1, -.5, 0, 1.2, 2.5, 3.1, 4, 5)]]
    #
    # divs = [[(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #         (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #         (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #         (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)],
    #         [(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #          (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #          (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #          (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)][::-1],
    #         [(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #          (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #          (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #          (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)]]
    #
    # smooth_lower_sums_op(divs, foos, (1280, 720), .25)

    # decompose_shape((1280, 720), .25)
    # triple_densing(lambda x: int(x*math.log(x + 3) + 1), 15, 3, FHD)
    absolute_value(lambda x: -x*(x-2)*(x+2)*(x-5)*(x+5)*math.exp(-abs(x))*math.sin(x)/12 - 1, speed=.1, resolution=FHD)