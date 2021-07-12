import basic_func
import objects


def epoch_frame(target_func, approximators_list, generate_png=False):
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    target_function = objects.Function(target_func)
    approximators = [objects.Function(foo) for foo in approximators_list]

    settings_target_function = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'white'
    }
    settings_approximators = {
        'sampling rate': 3,
        'thickness': 4,
        'blur': 2,
        'color': 'light gray'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 1,
        'color': 'white'
    }

    frame.add_axis_surface(x_bounds=(0, 1), y_bounds=(-2, 2))
    frame.add_axes(settings_axes, x_only=True)
    for foo in approximators:
        frame.blit_parametric_object(foo, settings_approximators)
    frame.blit_parametric_object(target_function, settings_target_function)
    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'frame.png')
    return frame


def make_film(target_func, epochs, filename='genetic.mp4'):
    video = basic_func.Film(5, (1280, 720))
    for i, epoch in enumerate(epochs):
        video.add_frame(epoch_frame(target_func, epoch))
        print(i)
    video.render(filename)

