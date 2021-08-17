import math
# import functools
# import sys

import numpy as np

import src.anamator.basic_func as basic_func
import src.anamator.objects as objects

from random import uniform, randint


def partition_fractal(no_of_points, no_of_fixed):
    fixed = [uniform(0, .4) for _ in range(no_of_fixed)] + [uniform(.6, 1) for _ in range(no_of_fixed)]
    random_points = [uniform(0, .4) for _ in range(no_of_points - no_of_fixed)] + [uniform(.6, 1) for _ in range(no_of_points - no_of_fixed)]
    random_points.sort()
    fixed.sort()
    before_zoom = [0.45+p/10 for p in fixed]
    for_zoom = [0.45+p/10 for p in random_points]
    return fixed, random_points


def dirichlet_function_but_enhanced(fixed, random_points):
    values_fixed = [randint(0, 1) for _ in range(len(fixed))]
    values_random = [randint(0, 1) for _ in range(len(random_points))]
    return values_fixed, values_random


def frame_gen(no_of_points, no_of_fixed):
    fixed, random_points = partition_fractal(no_of_points, no_of_fixed)
    values_fixed, values_random = dirichlet_function_but_enhanced(fixed, random_points)
    before_zoom = [0.45 + p / 10 for p in fixed]
    # for_zoom = [0.45 + p / 10 for p in random_points]

    centers = list(zip(fixed + random_points + before_zoom, values_fixed + values_random + values_fixed))
    before_zoom_points = [objects.BitmapDisk(3, 'white', 1) for _ in centers]

    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }

    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 3,
        'color': 'gray',
        'blur kernel': 'box'
    }

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    frame.add_axis_surface(x_bounds=(-1.1, 1.5), y_bounds=(-.3, 2))
    frame.blit_axes(settings_axes, x_only=False)
    frame.axis_surface.blit_distinct_bitmap_objects(centers, before_zoom_points, settings_axes)
    frame.blit_axis_surface()
    frame.generate_png('dirichlet.png')

    return frame

def animate_zoom():
    pass

if __name__ == '__main__':
    # print(random_points(30))
    frame_gen(5,2)
    print('dupa')