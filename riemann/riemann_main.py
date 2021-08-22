import math
import functools
import sys
import copy

import numpy as np

import src.anamator.basic_func as basic_func
import src.anamator.objects as objects


HD = (1280, 720)
FHD = (1920, 1080)
SHIT = (640, 480)


def linear(a=1):
    return lambda x: a*x


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
        'color': 'thistle'
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


def smooth_animate_divisions(divisions, resolution, speed, bigger_ab_at=2, filename='seq.mp4', radius_foo=1):
    filename = f'foo{radius_foo}' + filename
    init_x_length = .025
    settings_axes = {
        'sampling rate': 1,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }
    settings_bounds = {
        'sampling rate': 1,
        'thickness': 2,
        'blur': 0,
        'color': 'gray'
    }
    inverted_divisions = divisions[::-1]

    def radius(x):
        if x <= 3/4:
            return int(12*x)
        if x <= 1:
            return int(-16*x+21)
        else:
            return 5

    def big_radius2(x):
        if x <= 1/4:
            return int(60 * x + 5)
        if x <= 1/2:
            return int(-40 * x + 30)
        if x <= 3/4:
            return int(20*x)
        return int(-24*x + 35)

    def big_radius(x):
        if x <= 1/4:
            return int(40 * x + 5)
        if x <= 1/2:
            return int(-20 * x + 20)
        if x <= 3/4:
            return int(40*x - 20)
        return int(-44*x + 53)

    def big_radius3(x):
        if x <= 1/2:
            return int(30 * x + 5)
        return int(-22 * x + 31)

    def generate_division_frame(division, grid_settings, axis_settings, bounds_settings,
                                x_length=init_x_length, r=5, points=False, bigger_ab=False,
                                ab_radius=9, ab_color='white'):
        x_bounds = (division[0] - .1, division[-1] + .1)
        y_bounds = (-2, 2)
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        first_dash = objects.ParametricObject(make_const(divisions[0][0]), lambda t: t)
        second_dash = objects.ParametricObject(make_const(divisions[0][-1]), lambda t: t)
        frame.axis_surface.blit_dashed_curve(first_dash, 40, 200, bounds_settings, y_bounds, queue=True)
        frame.axis_surface.blit_dashed_curve(second_dash, 40, 200, bounds_settings, y_bounds, queue=False)
        if not points:
            lines = [objects.ParametricObject(make_const(point), lambda t: t, [-x_length, x_length])
                     for point in division]
            grid = functools.reduce(lambda x, y: x.stack_parametric_objects(y), lines)
            frame.axis_surface.blit_parametric_object(grid, grid_settings, interval_of_param=(-x_length, (2 * len(lines) - 1) * x_length))
        elif not bigger_ab:
            points = [objects.BitmapDisk(r, 'white', 1) for point in division]
            centers = list(map(lambda x: (x, 0), division))
            frame.axis_surface.blit_distinct_bitmap_objects(centers, points, settings_bounds)
        else:
            points = [objects.BitmapDisk(r, 'white', 1) for point in division[1:-1]]
            points = [objects.BitmapDisk(ab_radius, ab_color, 1)] + points + [objects.BitmapDisk(ab_radius, ab_color, 1)]
            centers = list(map(lambda x: (x, 0), division))
            frame.axis_surface.blit_distinct_bitmap_objects(centers, points, settings_bounds)
        frame.blit_axes(axis_settings, x_only=True)
        frame.blit_axis_surface()
        return frame

    def generate_frame(t):
        if t <= len(divisions) - 1:

            return generate_division_frame(basic_func.SingleAnimation.blend_lists(divisions, t), settings_axes,
                                           settings_axes, settings_bounds)
        if t <= len(divisions) - .5:
            t2 = t - len(divisions) + 1
            return generate_division_frame(divisions[-1], settings_axes, settings_axes, settings_bounds,
                                           x_length=(.5-t2)*init_x_length)
        if t <= len(divisions):
            t3 = 2*(t - len(divisions) + .5)
            return generate_division_frame(divisions[-1], settings_axes, settings_axes, settings_bounds, x_length=0,
                                           r=radius(t3), points=True)
        if t <= len(divisions) + bigger_ab_at:
            t4 = t - len(divisions)
            return generate_division_frame(basic_func.SingleAnimation.blend_lists(inverted_divisions, t4),
                                           settings_axes, settings_axes, settings_bounds, x_length=0,
                                           r=5, points=True)
        if t <= len(divisions) + bigger_ab_at + 1:
            t4 = t - len(divisions)
            t5 = t - len(divisions) - bigger_ab_at
            if radius_foo == 1:
                return generate_division_frame(basic_func.SingleAnimation.blend_lists(inverted_divisions, t4),
                                               settings_axes, settings_axes, settings_bounds, x_length=0,
                                               r=5, points=True, bigger_ab=True, ab_radius=big_radius(t5))
            if radius_foo == 2:
                return generate_division_frame(basic_func.SingleAnimation.blend_lists(inverted_divisions, t4),
                                               settings_axes, settings_axes, settings_bounds, x_length=0,
                                               r=5, points=True, bigger_ab=True, ab_radius=big_radius2(t5))
            return generate_division_frame(basic_func.SingleAnimation.blend_lists(inverted_divisions, t4),
                                           settings_axes, settings_axes, settings_bounds, x_length=0,
                                           r=5, points=True, bigger_ab=True, ab_radius=big_radius3(t5))

        t6 = t - len(divisions)
        return generate_division_frame(basic_func.SingleAnimation.blend_lists(inverted_divisions, t6),
                                       settings_axes, settings_axes, settings_bounds, x_length=0,
                                       r=5, points=True, bigger_ab=True, ab_radius=11)

    differential = objects.PredefinedSettings.exp_differential
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 2*len(divisions) - 1
    }
    animation.render(filename, settings, save_ram=True, id_='div', speed=speed)


def smooth_lower_sums_op(divisions, foos, resolution, speed, filename='op_lower_sums.mp4'):
    foo_settings = objects.PredefinedSettings.t2b0white
    axis_settings = objects.PredefinedSettings.t2b0white
    blended_rec_settings = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'magic mint'
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
        # t0 = min(math.floor(t) + 2*(t - math.floor(t)), math.floor(t) + 1)
        t0 = min(2*(t - math.floor(t)), 1)
        t1 = max(math.floor(t), math.floor(t) + 2*(t - math.floor(t))-1)

        division = basic_func.SingleAnimation.blend_lists(divs, t1)
        func = basic_func.SingleAnimation.blend_functions(foos, t1)
        x_bounds = (division[0] - 1, division[-1] + 1)
        y_bounds = (-1, 3)
        frame = basic_func.OneAxisFrame(resolution, 'black', 5, 5)
        frame.add_axis_surface(x_bounds, y_bounds)

        for interval in zip(division[:-1], division[1:]):

            level = objects.Function(const=t0*find_inf(func, interval) + (1-t0)*find_sup(func, interval)) \
                if math.floor(t) % 2 == 0 \
                else objects.Function(const=(1-t0)*find_inf(func, interval) + t0*find_sup(func, interval))
            level.add_bounds(interval)
            blended_rec = objects.FilledObject(level, objects.Function(const=0), interval)
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

    differential = objects.PredefinedSettings.slow_differential
    animation = basic_func.SingleAnimation(generate_frame, differential)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': len(divs) - 1
    }
    animation.render(filename, settings, save_ram=True, id_='op', speed=speed)


def smooth_animate_change_to_recs(division, foo, resolution, speed, filename='change_to_rec.mp4'):
    foo_settings = {
            'sampling rate': 3,
            'thickness': 7,
            'blur': 3,
            'color': 'white'
        }
    axis_settings = objects.PredefinedSettings.t5b2white
    blended_settings = {
            'sampling rate': 3,
            'thickness': 7,
            'blur': 3,
            'color': 'vivid tangerine'
        }
    blended_rec_settings = {
            'sampling rate': 3,
            'thickness': 0,
            'blur': 0,
            'color': 'vivid tangerine'
        }
    dash_settings = {
            'sampling rate': 10,
            'thickness': 2,
            'blur': 0,
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


def decompose_shape(resolution=FHD, speed=.1, filename='decompose.mp4'):
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
        'sampling rate': 10,
        'thickness': 0,
        'blur': 0,
        'color': 'red pastel 1'
    }

    def generate_frame(t):
        if t <= 1:
            t2 = 0
            t3 = 0
            y_bounds = (-25, 30)
            x_bounds = (-45, 20)
        elif t <= 2:
            t2 = t - 1
            t = 1
            t3 = 0
            x_bounds = (-45*(1-t2) + 1*t2, 20*(1-t2)+18.55*t2)
            y_bounds = (-25*(1-t2) + 5*t2, 30*(1-t2)+20*t2)
        else:
            t3 = t - 2
            t = 1
            t2 = 1
            x_bounds = (1, 18.55)
            y_bounds = (5, 20)

        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        # frame.blit_axes(axis_settings, x_only=False)

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
        frame.axis_surface.blit_filled_queue()

        # if t2 > .0001:
        #     ceil = objects.FilledObject(objects.Function(const=20), objects.Function(const=30), x_bounds)
        #     right_wall = objects.FilledObject(objects.Function(const=5), objects.Function(const=20), (-45, 1))
        #     left_wall = objects.FilledObject(objects.Function(const=5), objects.Function(const=20), (18.55, 20))
        #     floor = objects.FilledObject(objects.Function(const=-25), objects.Function(const=5), x_bounds)
        #     settings_walls = {
        #         'sampling rate': 1,
        #         'thickness': 0,
        #         'blur': 0,
        #         'color': (0, 0, 0, t2)
        #     }
        #     frame.axis_surface.blit_filled_object(ceil, settings_walls, queue=True)
        #     frame.axis_surface.blit_filled_object(right_wall, settings_walls, queue=True)
        #     frame.axis_surface.blit_filled_object(left_wall, settings_walls, queue=True)
        #     frame.axis_surface.blit_filled_object(floor, settings_walls, queue=True)
        #     frame.axis_surface.blit_filled_queue()

        if t3 > .0001:
            x_axis = objects.Function(const=10)
            y_axis = objects.ParametricObject(lambda x: 9.775, lambda y: y)
            frame.axis_surface.blit_parametric_object(y_axis, axis_settings, interval_of_param=(5, 5+t3*15), queue=True)
            frame.axis_surface.blit_parametric_object(x_axis, axis_settings, interval_of_param=(1, 1+t3*17.55), queue=True)
            frame.axis_surface.blit_parametric_queue()

        if t3 > .5:
            t4 = 2*(t3-.5)

            def thickness(x):
                if x <= .5:
                    return int(40*x)
                return int(-26*x + 28)

            settings_dash = {
                'sampling rate': 1,
                'thickness': thickness(t4),
                'blur': 0,
                'color': 'gray'
            }

            dash1 = objects.ParametricObject(lambda x: 4, lambda y: y)
            dash2 = objects.ParametricObject(lambda x: 15.55, lambda y: y)
            frame.axis_surface.blit_dashed_curve(dash1, 40, 50, settings_dash, queue=True)
            frame.axis_surface.blit_dashed_curve(dash2, 40, 50, settings_dash, queue=False)

        frame.blit_axis_surface()
        return frame

    differential = objects.PredefinedSettings.exp_differential
    animation = basic_func.SingleAnimation(generate_frame,
                                           basic_func.normalize_function(basic_func.make_periodic(differential)))
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 3
    }
    animation.render(filename, settings, save_ram=True, id_='op', speed=speed, precision=2000)


def get_triple_frame(division, resolution=HD, second_color='red pastel 2'):
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
        'color': second_color
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


def no_frame():
    resolution = (1920, 1080)
    function = lambda x: (abs(x) - abs(x - 1) + x ** 2 / 15) / 2 + 1.2
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
        'color': 'green 1'
    }
    settings_rec_2 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'green 2'
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
    frame.generate_png('no_frame.png')


def gradient4d(resolution=FHD, filename='gradient.mp4', no_frames=50, fps=5):

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
    settings_rec_1 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'red pastel 1'
    }

    integral = objects.FilledObject(objects.Function(const=0), func, area_bounds)
    frame.axis_surface_middle.blit_filled_object(integral, settings_rec_1, x_bounds, queue=False)
    frame.axis_surface_right.blit_filled_object(integral, settings_rec_1, x_bounds, queue=False)
    frame.axis_surface_left.blit_filled_object(integral, settings_rec_1, x_bounds, queue=False)

    frame.axis_surface_left.blit_parametric_object(func, settings_function)
    frame.axis_surface_right.blit_parametric_object(func, settings_function)
    frame.axis_surface_middle.blit_parametric_object(func, settings_function)

    frame.axis_surface_left.blit_axes(settings_axes, x_only=True)
    frame.axis_surface_right.blit_axes(settings_axes, x_only=True)
    frame.axis_surface_middle.blit_axes(settings_axes, x_only=True)

    frame.blit_axis_surfaces()
    # frame.blit_frames_around_axis_surfaces(settings_frames, x_inner_bounds=20, y_inner_bounds=20)

    densing_foo = lambda x: int(math.log(3*x+1)*x + 2)
    area_bounds = (-4, 4)
    film = basic_func.Film(fps, resolution)
    for i in range(no_frames):
        division = np.linspace(*area_bounds, densing_foo(i))
        film.add_frame(get_triple_frame(division, resolution,
                                        second_color=objects.ColorParser.blend('red pastel 2',
                                                                               'red pastel 1', i/no_frames)),
                       save_ram=True)
        print(f'{i + 1}/{no_frames} ({int(100 * (i + 1) / no_frames)}%)')
    for i in range(3*fps):
        film.add_frame(frame, save_ram=True)
    film.render(filename, save_ram=True)


def absolute_value(func, speed, resolution, filename='abs.mp4', id_='abs'):
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
        one_sign_intervals = blended_function.one_sign_intervals(area_bounds, precision=2000)
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

    animator = basic_func.FunctionSequenceAnimation((func, lambda t: -abs(func(t))),
                                                    objects.PredefinedSettings.fast_differential, gen_frame)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 1
    }
    animator.render(filename, settings, True, id_, speed=speed)


def intervals_into_divisions(division, speed, resolution, filename='intervals_into_points.mp4',
                             id_='intervals', slow=False):
    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 0,
        'color': 'white',
        'blur kernel': 'none'
    }

    def radius(x):
        if x <= 3/4:
            return int(16*x)
        if x <= 1:
            return int(-16*x+24)
        else:
            return 8

    def frame_gen(t):
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface((0, 1), (-1, 1))
        points = [objects.BitmapDisk(radius(t), 'white', 1) for _ in division]
        frame.axis_surface.blit_distinct_bitmap_objects(list(map(lambda tt: (tt, 0), division)), points,
                                                        settings_point_interior)

        if t > 1:
            for interval in zip(division[:-1], division[1:]):
                a, b = interval
                mean = (a+b)/2
                t_ = t - 1
                if not math.isclose(t_, 0, abs_tol=.01):
                    frame.axis_surface.blit_parametric_object(objects.Function(const=0),
                                                              objects.PredefinedSettings.fhd_foo,
                                                              (t_*a + (1-t_)*mean, t_*b + (1-t_)*mean), queue=True)
            frame.axis_surface.blit_parametric_queue()
        frame.blit_axis_surface()
        return frame

    animator = basic_func.SingleAnimation(frame_gen, objects.PredefinedSettings.slow_differential) \
        if slow else basic_func.SingleAnimation(frame_gen, objects.PredefinedSettings.fast_differential)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 2
    }
    animator.render(filename, settings, True, id_, speed=speed)


def bold(interval, bolded_interval, foo, x_bounds=(-1, 1), resolution=FHD, filename='bold.mp4', id_='bold', speed=1,
         dark_color='blue bell', light_color='thistle'):
    filename = dark_color + '_' + light_color + '_' + filename
    downing_settings = {
        'sampling rate': 1,
        'thickness': 5,
        'blur': 0,
        'color': dark_color
    }
    dark_rec_settings = {
        'sampling rate': 1,
        'thickness': 0,
        'blur': 0,
        'color': dark_color
    }
    foo_settings = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 3,
        'color': 'white'
    }
    axis_settings = {
        'sampling rate': 1,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }
    dash_settings = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'gray'
    }
    fill_downing_settings = {
        'sampling rate': 1,
        'thickness': 0,
        'blur': 0,
        'color': light_color
    }
    lines_settings = {
        'sampling rate': 1,
        'thickness': 5,
        'blur': 0,
        'color': 'gray'
    }
    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 0,
        'blur': 0,
        'color': 'white',
        'blur kernel': 'box'
    }

    def radius(x):
        if x <= 3 / 4:
            return int(16 * x)
        if x <= 1:
            return int(-16 * x + 24)
        else:
            return 8

    def thickness(x):
        if x <= 3 / 4:
            return int(25 * x)
        if x <= 1:
            return int(-40 * x + 50)
        else:
            return 10

    def generate_frame(t):
        t1 = min(t, 1)
        t2 = min(t - 1, 1)
        t3 = min(t - 2, 1)
        t4 = min(t - 3, 1)
        t5 = min(t - 4, 1)

        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        func = objects.Function(foo)
        argmin = func.argmin(bolded_interval)
        argmax = func.argmax(bolded_interval)
        y_bounds = (-1, func.sup(x_bounds)+1)
        frame.add_axis_surface(x_bounds, y_bounds)
        frame.blit_axes(axis_settings)

        for part in zip(interval[:-1], interval[1:]):
            downing = objects.Function(basic_func.SingleAnimation.blend_functions((foo, make_const(func.inf(part))),
                                                                                  t1))
            frame.axis_surface.blit_parametric_object(downing, settings=downing_settings,
                                                      queue=True, interval_of_param=part)
            filling = objects.FilledObject(downing, objects.Function(basic_func.SingleAnimation.blend_functions((copy.copy(downing), lambda h: 0), t1)), part)
            frame.axis_surface.blit_filled_object(filling, fill_downing_settings, interval_of_param=part, queue=True)

        frame.axis_surface.blit_filled_queue()
        frame.axis_surface.blit_parametric_queue()

        if t > 3 and not math.isclose(t, 3):
            rec = objects.FilledObject(objects.Function(const=func(argmin)),
                                       objects.Function(const=(1-t4)*func(argmin) + t4*func(argmax)), bolded_interval)
            frame.axis_surface.blit_filled_object(rec, dark_rec_settings)

        for x in interval:
            line = objects.ParametricObject(make_const(x), lambda y: y, y_bounds)
            frame.axis_surface.blit_dashed_curve(line, 40, settings=dash_settings,
                                                 interval_of_param=y_bounds, queue=True)
        frame.axis_surface.blit_parametric_queue()

        if t > 2 and not math.isclose(t, 2):

            const_settings = {
                'sampling rate': 3,
                'thickness': thickness(t3),
                'blur': 0,
                'color': dark_color
            }
            bolded_const = objects.Function(const=func.inf(bolded_interval))
            bolded_const_int = [t3*bolded_interval[0] + (1-t3)*argmin, t3*bolded_interval[1] + (1-t3)*argmin]
            frame.axis_surface.blit_parametric_object(bolded_const, const_settings, bolded_const_int)

        if t > 4 and not math.isclose(t, 4):
            const_settings = {
                'sampling rate': 3,
                'thickness': thickness(t5),
                'blur': 0,
                'color': light_color
            }
            bolded_const = objects.Function(const=func.sup(bolded_interval))
            bolded_const_int = [t5*bolded_interval[0] + (1-t5)*argmax, t5*bolded_interval[1] + (1-t5)*argmax]
            frame.axis_surface.blit_parametric_object(bolded_const, const_settings, bolded_const_int)

        frame.blit_parametric_object(func, settings=foo_settings)

        if t > 1 and not math.isclose(t, 1):
            color = (0, 0, 0, 3*t2/4)
            gray_wall1 = objects.FilledObject(objects.Function(const=y_bounds[0]), objects.Function(const=y_bounds[1]),
                                              (x_bounds[0], bolded_interval[0]))
            gray_wall2 = objects.FilledObject(objects.Function(const=y_bounds[0]), objects.Function(const=y_bounds[1]),
                                              (bolded_interval[1], x_bounds[1]))
            wall_settings = {
                'sampling rate': 2,
                'thickness': 0,
                'blur': 0,
                'color': color
            }
            frame.axis_surface.blit_filled_object(gray_wall1, wall_settings, queue=True)
            frame.axis_surface.blit_filled_object(gray_wall2, wall_settings, queue=True)
            frame.axis_surface.blit_filled_queue()

            middle = np.mean(y_bounds)
            line_bounds = t2 * np.array(y_bounds) + (1 - t2) * np.array([middle, middle])
            line1 = objects.ParametricObject(make_const(bolded_interval[0]), lambda y: y, line_bounds)
            line2 = objects.ParametricObject(make_const(bolded_interval[1]), lambda y: y, line_bounds)
            frame.axis_surface.blit_parametric_object(line1, lines_settings, queue=True)
            frame.axis_surface.blit_parametric_object(line2, lines_settings, queue=True)
            frame.axis_surface.blit_parametric_queue()

        if t > 2 and not math.isclose(t, 2):
            circle = objects.BitmapDisk(int(.66*radius(t3))+2, dark_color, 1)
            dot = objects.BitmapDisk(int(.66*radius(t3)), light_color, 1)
            frame.axis_surface.blit_bitmap_object((argmin, func(argmin)), circle,
                                                  settings_point_interior, surface_coordinates=True)
            frame.axis_surface.blit_bitmap_object((argmin, func(argmin)), dot,
                                                  settings_point_interior, surface_coordinates=True)

        if t > 4 and not math.isclose(t, 4):
            circle = objects.BitmapDisk(int(.66*radius(t5))+2, light_color, 1)
            dot = objects.BitmapDisk(int(.66*radius(t5)), dark_color, 1)
            frame.axis_surface.blit_bitmap_object((argmax, func(argmax)), circle,
                                                  settings_point_interior, surface_coordinates=True)
            frame.axis_surface.blit_bitmap_object((argmax, func(argmax)), dot,
                                                  settings_point_interior, surface_coordinates=True)
        frame.blit_axis_surface()
        return frame

    animator = basic_func.SingleAnimation(generate_frame, objects.PredefinedSettings.exp_differential)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 5
    }
    animator.render(filename, settings, save_ram=True, id_=id_, speed=speed, read_only=False)


def hill(speed, resolution, filename='hill.mp4'):
    seq = [lambda x: 0,
           lambda x: 8*math.exp(-(x-3)**2),
           lambda x: 8*math.exp(-(x-3)**2) - 3*math.exp(-(x-5.3)**4)]

    settings_rec_positive = {
        'sampling rate': 2,
        'thickness': 0,
        'blur': 3,
        'color': 'tea green'
    }

    def fame_generator(foo):
        frame = basic_func.OneAxisFrame(resolution, 'black', 0, 0)
        frame.add_axis_surface((0, 8), (-5, 10))
        ground = objects.FilledObject(objects.Function(const=-5), objects.Function(foo), (0, 8))
        frame.axis_surface.blit_filled_object(ground, settings_rec_positive)
        frame.blit_axis_surface()
        return frame

    animator = basic_func.FunctionSequenceAnimation(seq, objects.PredefinedSettings.fast_differential, fame_generator)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 2
    }
    animator.render(filename, settings, save_ram=True, id_='koparaexp', speed=speed)


NO_RANDOM = 50
NO_NICE = 30


def plum_dirichlet(filename='plum_dirichlet_6px.mp4', speed=1, resolution=FHD, save=False):
    def make_exp_diff(x0):
        return basic_func.normalize_function(lambda x: math.exp(-70*(x-x0)**2), interval=(0, 2))

    differentials = [make_exp_diff(.5)]\
                    + [make_exp_diff(k) for k in np.linspace(1, 1.75, NO_NICE)]*2

    def radius(x):
        if x <= 3/4:
            return int(6+6*x)
        if x <= 1:
            return int(-18*x+24)
        else:
            return 6

    settings_dots = {
        'blur': 3,
        'blur kernel': 'box'
    }

    x_min = -.25
    x_max = 1.25

    random_1_points = [(min(max(np.random.normal(k, .1), x_min), x_max), .3) for k in np.linspace(x_min, x_max, NO_RANDOM)]
    random_0_points = [(min(max(np.random.normal(k, .1), x_min), x_max), 0) for k in np.linspace(x_min, x_max, NO_RANDOM)]
    random_0_points += [(min(max(np.random.normal(k, .1), x_min), 0), 0) for k in np.linspace(x_min, 0, NO_NICE//2)]
    random_0_points += [(min(max(np.random.normal(k, .1), 1), x_max), 0) for k in np.linspace(1, x_max, NO_NICE//2)]
    random_1_points += [(min(max(np.random.normal(k, .1), x_min), 0), .3) for k in np.linspace(x_min, 0, NO_NICE//2)]
    random_1_points += [(min(max(np.random.normal(k, .1), 1), x_max), .3) for k in np.linspace(1, x_max, NO_NICE//2)]

    nice_1_points = [(k, .3) for k in np.linspace(0, 1, NO_NICE)]
    nice_0_points = [(k, 0) for k in np.linspace(0, 1, NO_NICE)]

    def generate_frame(*t_list):
        t0 = t_list[0]
        color = (0, 0, 0, 3 * min(t0, 1) / 4)

        frame = basic_func.OneAxisFrame(resolution)
        frame.add_axis_surface((x_min, x_max), (-.3, .8))

        nice_dots = [objects.BitmapDisk(radius(t), 'white', 1) for t in t_list[1:]]
        random_dot = objects.BitmapDisk(6, 'white', 1)

        frame.blit_axes(objects.PredefinedSettings.fhd_axis, x_only=True)

        for dot, center in zip(nice_dots, nice_1_points):
            frame.axis_surface.blit_bitmap_object(center, dot, settings_dots)
        for dot, center in zip(nice_dots, nice_0_points):
            frame.axis_surface.blit_bitmap_object(center, dot, settings_dots)
        for center in random_1_points + random_0_points:
            frame.axis_surface.blit_bitmap_object(center, random_dot, settings_dots)

        gray_wall1 = objects.FilledObject(objects.Function(const=-.3), objects.Function(const=.8), (x_min, 0))
        gray_wall2 = objects.FilledObject(objects.Function(const=-.3), objects.Function(const=.8), (1, x_max))

        wall_settings = {
            'sampling rate': 2,
            'thickness': 0,
            'blur': 0,
            'color': color
        }
        frame.axis_surface.blit_filled_object(gray_wall1, wall_settings, queue=True, interval_of_param=(x_min, 0))
        frame.axis_surface.blit_filled_object(gray_wall2, wall_settings, queue=True, interval_of_param=(1, x_max))
        frame.axis_surface.blit_filled_queue()

        line_0 = objects.ParametricObject(lambda x: 0, lambda y: y)
        line_1 = objects.ParametricObject(lambda x: 1, lambda y: y)
        bounds = (-.55 * min(t0, 1) + .25, .25 + .55 * min(t0, 1))
        if bounds[1] - bounds[0] > .01:
            frame.axis_surface.blit_dashed_curve(line_0, 40, 50, objects.PredefinedSettings.t2b0gray,
                                                 interval_of_param=bounds, queue=True)
            frame.axis_surface.blit_dashed_curve(line_1, 40, 50, objects.PredefinedSettings.t2b0gray,
                                             interval_of_param=bounds, queue=False)

        frame.blit_axis_surface()
        if save:
            frame.generate_png('test_speedrun.png')
        return frame

    settings = {
        'fps': 24,
        'duration': 2,
        'resolution': resolution
    }

    animator = basic_func.MultiDifferentialAnimation(generate_frame, *differentials)
    animator.render(filename, settings, save_ram=True, speed=speed, id_='plum_dirichlet')


def flying_recs(filename='flying_recs.mp4', resolution=FHD, speed=1):
    init_x_length = .025
    settings_axes = {
        'sampling rate': 1,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }
    settings_bounds = {
        'sampling rate': 1,
        'thickness': 2,
        'blur': 0,
        'color': 'gray'
    }
    settings_recs1 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'baby powder'
    }
    settings_recs2 = {
        'sampling rate': 3,
        'thickness': 0,
        'blur': 0,
        'color': 'shimmering blush'
    }
    division = [0, .08, .15, .24, .36, .45, .6, .75, .88, .95, 1]
    x_bounds = (division[0] - .1, division[-1] + .1)
    y_bounds = (-2, 2)

    rec_size = [1.5, .3, -.4, .7, -1.6, -1.8, .2, -.9, .1, -1.2]

    def gen_frame(t):
        frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
        frame.add_axis_surface(x_bounds, y_bounds)
        recs = [objects.FilledObject(objects.Function(const=t*h), objects.Function(const=0), interval)
                for h, interval in zip(rec_size, zip(division[:-1], division[1:]))]

        for obj in recs[::2]:
            frame.axis_surface.blit_filled_object(obj, settings_recs1, queue=True)
        frame.axis_surface.blit_filled_queue()

        for obj in recs[1::2]:
            frame.axis_surface.blit_filled_object(obj, settings_recs2, queue=True)
        frame.axis_surface.blit_filled_queue()

        first_dash = objects.ParametricObject(make_const(division[0]), linear())
        second_dash = objects.ParametricObject(make_const(division[-1]), linear())
        frame.axis_surface.blit_dashed_curve(first_dash, 40, 200, settings_bounds, y_bounds, queue=True)
        frame.axis_surface.blit_dashed_curve(second_dash, 40, 200, settings_bounds, y_bounds, queue=False)
        frame.blit_axes(settings_axes, x_only=True)
        lines = [objects.ParametricObject(make_const(point), lambda x: x, [-init_x_length, init_x_length])
                 for point in division]
        grid = functools.reduce(lambda x, y: x.stack_parametric_objects(y), lines)
        frame.axis_surface.blit_parametric_object(grid, settings_bounds,
                                                  interval_of_param=(-init_x_length, (2*len(lines) - 1)*init_x_length))
        frame.blit_axis_surface()
        return frame

    def differential(t):
        return math.sin(1.5*math.pi*t)*(1 + int(math.sin(1.5*math.pi*t) < 0)) if t < 2 \
            else math.sin(3*math.pi*t)*(1 + int(math.sin(3*math.pi*t) < 0))/2

    differential = objects.normalize_function(differential, (0, .66))

    animator = basic_func.SingleAnimation(gen_frame, differential)
    # animator = basic_func.SingleAnimation(gen_frame, objects.normalize_function(differential, (0, .5)))

    settings = {
        'fps': 24,
        'duration': 4,
        'resolution': resolution
    }

    animator.render(filename, settings, speed=speed)


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
    # smooth_animate_changing_integral(func_seq, int_seq, FHD)
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
    # #                               'change_to_rec_dense.mp4')
    # # #
    # foos = [lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 1.2, lambda x: x*(x-2)*(x+2)*(x-5)*(x+5)*math.exp(-abs(x))*math.sin(x)/12+1,
    #         lambda x: (abs(x) - abs(x-1) + x**2/15)/2 + 1.2]
    #
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
    # #
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
    # smooth_lower_sums_op(divs, foos, FHD, 1)

    # decompose_shape((1280, 720), .25)
    # triple_densing(lambda x: int(x*math.log(x + 3) + 1), 15, 3, FHD)
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x/2)*math.sin(5*x)**2+1.2,
    #      resolution=FHD, speed=.4)
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='nyanza', dark_color='pistachio')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #          resolution=FHD, speed=.4, light_color='columbia blue', dark_color='beau blue')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='beau blue', dark_color='cerulean frost')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='silver pink', dark_color='tuscany')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='tuscany', dark_color='eggplant')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='tea green', dark_color='eton blue')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='eton blue', dark_color='cadet gray')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.25, light_color='beige', dark_color='irresistible')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='banana mania', dark_color='grullo')
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.4, light_color='banana mania', dark_color='coyote brown')
    # decompose_shape(FHD, .1)
    plum_dirichlet(speed=.3, resolution=FHD)
