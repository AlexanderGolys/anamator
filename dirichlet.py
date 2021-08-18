import math

import numpy as np

import src.anamator.basic_func as basic_func
import src.anamator.objects as objects

from random import uniform, randint


def partition_fractal(no_of_points, no_of_fixed, interval_length):
    fixed = [uniform(0, .4*interval_length) for _ in range(no_of_fixed)] + [uniform(.6*interval_length, 1) for _ in range(no_of_fixed)]
    random_points = [uniform(0, .4*interval_length) for _ in range(no_of_points - no_of_fixed)] + [uniform(.6*interval_length, 1) for _ in range(no_of_points - no_of_fixed)]
    random_points.sort()
    fixed.sort()
    before_zoom = [0.45*interval_length+p/10 for p in fixed]
    for_zoom = [0.45*interval_length+p/10 for p in random_points]
    return fixed, random_points


def dirichlet_function_but_enhanced(fixed, random_points):
    values_fixed = [randint(0, 1) for _ in range(len(fixed))]
    values_random = [randint(0, 1) for _ in range(len(random_points))]
    return values_fixed, values_random


def split_object_list(centers, radius):

    too_close_centers = []
    i = 1
    while i < len(centers)-1:
        if centers[i][1] == centers[i + 1][1]:
            if abs(centers[i][0] - centers[i + 1][0]) < 2 * radius:
                print(centers[i], centers[i + 1])
                too_close_centers.append(centers[i + 1])
                del centers[i]
                if i == len(centers):
                    del centers[i]
            else:
                i += 1
        else:
            i += 1

    return [centers, too_close_centers]


def frame_gen(no_of_points, no_of_fixed, interval):
    fixed, random_points = partition_fractal(no_of_points, no_of_fixed, interval[1]-interval[0])
    values_fixed, values_random = dirichlet_function_but_enhanced(fixed, random_points)
    before_zoom = [0.45*(interval[1]-interval[0]) + p / 10*(interval[1]-interval[0]) for p in fixed]
    # for_zoom = [0.45 + p / 10 for p in random_points]

    centers = list(zip(fixed + random_points + before_zoom, values_fixed + values_random + values_fixed))

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

    frame = basic_func.OneAxisFrame((3840, 2160), 'black', 100, 100)
    frame.add_axis_surface(x_bounds=(interval[0]-.1, interval[1] + .1), y_bounds=(-.3, 2))
    frame.blit_axes(settings_axes, x_only=False)
    list_of_centers = split_object_list([frame.axis_surface.transform_to_pixel_coordinates(ctr) for ctr in centers], 8)
    before_zoom_points = [[objects.BitmapDisk(8, 'white', 1, padding=0) for _ in centers] for centers in list_of_centers]
    frame.axis_surface.blit_distinct_bitmap_objects(centers, before_zoom_points, settings_point_interior)
    # for pt in zip(centers, before_zoom_points):
    #     frame.axis_surface.blit_distinct_bitmap_objects([pt[0]], [pt[1]], settings_point_interior)
    frame.blit_axis_surface()
    frame.generate_png('dirichlet.png')

    return frame


def animate_zoom():
    pass

if __name__ == '__main__':
    # print(random_points(30))
    # frame_gen(15,5,(0,3))
    # print('dupa')
    ctrs = [(1,0), (1.5,0), (2,0), (3,0), (3.7,0),(5,0), (5.6,0), (6,0), (8,0)]
    r = .4
    print(split_object_list(ctrs,r))
