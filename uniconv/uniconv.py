import math

from animator import basic_func, objects
import numpy as np

limit_func = lambda x: 0.2

def func_seq(n):
    return lambda x: (x**n + 0.2)


def sup_metric(f, g, interval, m):
    d = 0.0
    partition = np.arange(interval[0], interval[1]+1/m, 1/m)
    # print(f,g)
    return np.amax(np.asarray([(d*(abs(f(x)-g(x))<d) + abs(f(x)-g(x))*(abs(f(x)-g(x))>d)) for x in partition]))

def epsilon(n, f, g, interval, m):
    return sup_metric(f, g(n), interval, m)
# eps_k = epsilon(k, limit_function, function_seq, interval, m)

def function_eps(limit_function, plus, eps):
    return lambda x: (limit_function)(x)+eps if plus else (limit_function)(x)-eps


def generate_frame(n, m, limit_function, function_seq, interval, generate_png=False):

    sine = objects.Function(lambda x: math.sin(x))
    eps = sup_metric(limit_function, function_seq, interval, m)
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    func = objects.Function(limit_function)
    func_seq = objects.Function(function_seq)
    eps_plus = objects.Function(function_eps(limit_function, True, eps))
    eps_minus = objects.Function(function_eps(limit_function, False, eps))

    settings_function = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'gray'
    }
    settings_sine = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'blue'
    }
    settings_function_seq = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'red'
    }
    settings_epsilon = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'green'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 1,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 2,
        'color': 'white'
    }

    frame.add_axis_surface(x_bounds=interval, y_bounds=(-3, 3))
    frame.blit_axes(settings_axes, x_only=False)
    frame.blit_parametric_object(func, settings_function)
    frame.blit_parametric_object(func_seq, settings_function_seq)
    frame.axis_surface.blit_parametric_object(eps_plus, settings_epsilon, queue=True)
    frame.axis_surface.blit_parametric_object(eps_minus, settings_epsilon, queue=True)
    frame.axis_surface.blit_parametric_queue()
    #frame.blit_parametric_object(eps_minus, settings_epsilon)
    frame.blit_parametric_object(sine, settings_sine)


    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'uniconv{n}.png')
    return frame

def render_video(n, m, limit_func, interval, start=1, filename='uniconv_test.mp4'):
    video = basic_func.Film(2, (1280, 720))
    for i in range(start, n):
        # video.add_frame(generate_frame(i, generate_png=False, foo=foo))
        # func_seq_i = lambda x: func_seq(i)
        video.add_frame(generate_frame(n, m, limit_func, func_seq(i), interval), save_ram=True)
        print(i)
    video.render(filename, save_ram=True)

if __name__ == '__main__':
    render_video(3, 1000, limit_func, (-0.2,0.75))