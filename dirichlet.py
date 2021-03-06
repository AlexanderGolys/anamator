import math
import numpy as np

import src.anamator.basic_func as basic_func
import src.anamator.objects as objects

from random import uniform, randint, seed

# (1) enhance random points
# (2) choose the equal amount of 1s and 0s for values


def dirichlet_function_but_enhanced(fixed, random_points):
    values_fixed = [.3*randint(0, 1) for _ in range(len(fixed))]
    values_random = [.3*randint(0, 1) for _ in range(len(random_points))]
    return values_fixed, values_random


def split_object_list(centers, radius, limit):

    too_close_centers = []

    for k, ctr in enumerate(centers[:-1]):
        tmp = centers[k+1]
        if ctr[1] == tmp[1] and abs(ctr[0] - tmp[0]) < 2 * radius:
            too_close_centers.append(tmp)

    centers = list(set(centers)-set(too_close_centers))
    centers.sort()
    complete = [centers, too_close_centers]

    if limit > 0:
        more_split = split_object_list(too_close_centers, radius, limit-1)
        if more_split is not None:
            if len(more_split) == 2:
                more_ctrs, even_more_ctrs = more_split
                if len(even_more_ctrs) > 0:
                    complete.pop()
                    complete.append(more_ctrs)
                    complete.append(even_more_ctrs)
                else:
                    return
    return complete


def partition_fractal(no_of_points, no_of_fixed, interval_length):
    fixed = [uniform(0, .4*interval_length) for _ in range(no_of_fixed)] + [uniform(.6*interval_length, interval_length) for _ in range(no_of_fixed)]
    random_points = [uniform(0, .4*interval_length) for _ in range(no_of_points)] + [uniform(.6*interval_length, interval_length) for _ in range(no_of_points)]
    random_points.sort()
    fixed.sort()

    return fixed, random_points


def single_frame(partition, values, interval, interval_but_smol, radius, indices, *t, resolution=(1280, 720)):

    def radii(x):
        if x <= 3/4:
            return int(2*radius*x)
        if x <= 1:
            return int(-2*radius*x+3*radius)
        else:
            return radius

    transform_to_smol = lambda x: .4 + x/5

    t_0 = t[0]
    t = t[1:]

    fixed, random_points = partition
    values_fixed, values_random = values
    before_zoom = [transform_to_smol(p) for p in fixed]
    for_zoom = [transform_to_smol(p) for p in before_zoom] + [transform_to_smol(p) for p in random_points]

    centers = list(zip(fixed + random_points + before_zoom, values_fixed + values_random + values_fixed))
    for_zoom_centers = list(zip(for_zoom, values_fixed + values_random))

    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }

    settings_point_interior = {
        'sampling rate': 1,
        'thickness': 1,
        'blur': 2,
        'color': 'gray',
        'blur kernel': 'box'
    }

    frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
    frame.add_axis_surface(x_bounds=basic_func.SingleAnimation.blend_lists([interval, interval_but_smol],t_0), y_bounds=(-.3, .8))
    # frame.axis_surface.blit_scale(objects.PredefinedSettings.t2b0white, x_interval=.05, x_length=.02)
    frame.blit_axes(settings_axes, x_only=False)
    list_of_centers = split_object_list([frame.axis_surface.transform_to_pixel_coordinates(ctr) for ctr in centers], 32, 0)
    before_zoom_points = [[objects.BitmapDisk(radius, 'white', 1, padding=0) for _ in centers] for centers in list_of_centers]
    for pts in zip(list_of_centers, before_zoom_points):
        frame.axis_surface.blit_distinct_bitmap_objects(pts[0], pts[1], settings_point_interior, surface_coordinates=False)

    list_of_for_zoom_centers = split_object_list([frame.axis_surface.transform_to_pixel_coordinates(ctr) for ctr in for_zoom_centers], 32,
                                        0)
    for_zoom_points = [objects.BitmapDisk(radii(t[i]), 'white', 1, padding=0) for i in indices]
    for pts in list_of_for_zoom_centers:
        frame.axis_surface.blit_distinct_bitmap_objects(pts, for_zoom_points[:len(pts)], settings_point_interior,
                                                        surface_coordinates=False)
        for_zoom_points = for_zoom_points[len(pts):]

    frame.blit_axis_surface()
    frame.generate_png('dirichlet.png')

    return frame


def one_dirichlet(no_of_points, no_of_fixed, radius, interval, resolution):
    partition = partition_fractal(no_of_points, no_of_fixed, interval[1]-interval[0])
    values = dirichlet_function_but_enhanced(partition[0], partition[1])
    fixed, random_points = partition
    values_fixed, values_random = values
    transform_to_smol = lambda x: .4 + x / 5
    before_zoom = [transform_to_smol(p) for p in fixed]
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
        'blur': 2,
        'color': 'gray',
        'blur kernel': 'box'
    }

    frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
    frame.add_axis_surface(x_bounds=(-.1, 1.1), y_bounds=(-.3, .8))
    frame.blit_axes(settings_axes, x_only=False)
    point_objects = [objects.BitmapDisk(radius, 'white', 1) for ctr in centers]

    for pt in zip(centers, point_objects):
        frame.axis_surface.blit_distinct_bitmap_objects([pt[0]], [pt[1]], settings_point_interior)

    frame.blit_axis_surface()
    frame.generate_png('one_dirichlet.png')


def single_frame_without_split(partition, values, interval, interval_but_smol, radius, indices, *t, resolution=(1280, 720)):

    # def radii(x):
    #     if x <= 3/4:
    #         return int(2*radius*x)
    #     if x <= 1:
    #         return int(-2*radius*x+3*radius)
    #     else:
    #         return radius

    def radii(x):
        if x <= 3/4:
            return int(16*x)
        if x <= 1:
            return int(-16*x+24)
        else:
            return 8

    transform_to_smol = lambda x: .4 + x/5

    t_0 = t[0]
    t = t[1:]

    fixed, random_points = partition
    values_fixed, values_random = values
    before_zoom = [transform_to_smol(p) for p in fixed]
    for_zoom = [transform_to_smol(p) for p in before_zoom] + [transform_to_smol(p) for p in random_points]

    centers = list(zip(fixed + random_points + before_zoom, values_fixed + values_random + values_fixed))
    for_zoom_centers = list(zip(for_zoom, values_fixed + values_random))

    settings_axes = {
        'sampling rate': 3,
        'thickness': 3,
        'blur': 0,
        'color': 'white'
    }

    settings_point_interior = {
        'blur': 3,
        'blur kernel': 'box'
    }

    frame = basic_func.OneAxisFrame(resolution, 'black', 100, 100)
    frame.add_axis_surface(x_bounds=basic_func.SingleAnimation.blend_lists([[interval[0], interval[1]], interval_but_smol],t_0), y_bounds=(-.3, .8))
    frame.blit_axes(objects.PredefinedSettings.fhd_axis, x_only=True)

    before_zoom_points = [objects.BitmapDisk(radius, 'white', 1) for ctr in centers]
    for_zoom_points = [objects.BitmapDisk(radii(t[i]), 'white', 1, padding=0) for i in indices]

    for pt in zip(centers, before_zoom_points):
        frame.axis_surface.blit_bitmap_object(pt[0], pt[1], settings_point_interior)

    for pt in zip(for_zoom_centers, for_zoom_points):
        frame.axis_surface.blit_bitmap_object(pt[0], pt[1], settings_point_interior)

    frame.axis_surface.blit_bitmap_object((.5, .3), objects.BitmapDisk(radius, 'white', 1), settings_point_interior)
    frame.axis_surface.blit_bitmap_object((.5, 0), objects.BitmapDisk(radius, 'white', 1), settings_point_interior)

    frame.blit_axis_surface()
    frame.generate_png('dirichlet.png')

    return frame


def animate_zoom(no_of_points, no_of_fixed, radius, interval, interval_but_smol, speed, resolution, filename='dirichlet_func_frac.mp4',
                             id_='dirichlet', slow=False):
    indices = [randint(0, 9) for _ in range(no_of_fixed + 2 * no_of_points)]
    partition = partition_fractal(no_of_points, no_of_fixed, interval[1]-interval[0])
    values = dirichlet_function_but_enhanced(partition[0], partition[1])

    def frame_gen(*t):
        return single_frame_without_split(partition, values, interval, interval_but_smol, radius, indices, *t, resolution=resolution)

    def make_exp_diff(x0):
        return basic_func.normalize_function(lambda x: math.exp(-200*(x-x0)**2))

    differentials = [lambda x: 1.02] + [make_exp_diff(t) for t in np.linspace(.25, .75, 10)]

    animator = basic_func.MultiDifferentialAnimation(frame_gen, *differentials)
    settings = {
        'fps': 24,
        'resolution': resolution,
        'duration': 1
    }
    animator.render(filename, settings, True, id_, speed=speed)


if __name__ == '__main__':
    # print(random_points(30))
    # single_frame(50,25,(0,3),3,0)
    print('dupa')
    seed(1)
    # ctrs = [(1,0), (1.5,0), (2,0), (3,0), (3.7,0),(5,0), (5.6,0), (6,0), (8,0), (8,1)]
    # r = .4
    # print(split_object_list(ctrs,r,5))
    animate_zoom(50, 25, 8, (0, 1), (.4, .6), .1, (1920, 1080))
    # one_dirichlet(50, 25, 7, (0, 1), (3840, 2160))
