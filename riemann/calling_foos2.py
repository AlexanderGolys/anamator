from riemann.riemann_main import *


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


if __name__ == '__main__':
    intervals_into_divisions([.1, .3, .4, .7, .9], 1, HD)
