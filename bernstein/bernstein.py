import math

from animator import basic_func, objects


def bernstein_basis(k, n):
    return lambda x: math.comb(n, k)*x**k*(1-x)**(n-k)


def bernstein(foo, n):
    return lambda x: sum([foo(k/n)*bernstein_basis(k, n)(x) for k in range(1, n+1)])


def generate_frame(n, generate_png=False, foo=lambda x: 0 if x == 0 else x*math.sin(1/x)):

    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    func = objects.Function(foo)
    func2 = objects.Function(bernstein(foo, n))

    settings_function = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'gray'
    }
    settings_function2 = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'white'
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

    frame.add_axis_surface(x_bounds=(0, 1), y_bounds=(-2, 2))
    frame.blit_axes(settings_axes, x_only=True)
    frame.blit_parametric_object(func, settings_function)
    frame.blit_parametric_object(func2, settings_function2)

    # frame.blit_x_grid(settings_grid, interval=.1, length=.01)
    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'xsin_bern{n}.png')
    return frame


def render_video(n, foo=lambda x: 0 if x == 0 else x*math.sin(1/x), start=0, filename='xsin_bernstein_hd.mp4',
                 save_ram=True):
    video = basic_func.Film(5, (1280, 720))
    for i in range(start, n):
        video.add_frame(generate_frame(i, generate_png=False, foo=foo), save_ram=save_ram)
        print(i)
    video.render(filename, save_ram=save_ram)


if __name__ == '__main__':
    # generate_frame(40, True)
    render_video(45, start=0, foo=lambda x: x**(1/2), filename='sqrt.mp4')

