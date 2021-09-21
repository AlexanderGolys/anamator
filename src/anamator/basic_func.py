import itertools
import functools
import math
import os
import copy
import deprecation

import numpy as np
import scipy.signal
import cv2

from PIL import Image

from . import objects
from .objects import normalize_function, make_periodic


DEBUG = True
DEBUG_SHORT = False


def debug(log, short=True):
    if DEBUG_SHORT or (DEBUG and not short):
        print(f'--- {log} ---')


def create_const(c):
    return lambda x: c


class Surface:
    """
    Single bitmap object.

    Attributes:
        res (tuple of ints):  Frame resolution in pixels.
        bitmap (np.array): Bitmap in RGBA.
    """

    def __init__(self, res):
        """
        Args:
            res (tuple of ints):  Frame resolution in pixels.
        """
        debug('surface init')
        self.res = res
        self.bitmap = np.zeros(self.res + (4,))

    def blit_surface(self, surface, ll_corner, ur_corner=None, blank=False):
        """
        Merge surfaces. It scales the surface if lower right corner is provided.

        Args:
            surface (Surface): Surface to be blitted.
            ll_corner (tuple of ints): Lower left corner in pixel coordinates.
            ur_corner (tuple of ints, optional): Upper right corner in pixel coordinates. If provided surface will
                be scaled to fill given box. If None, surface wil be blitted without scaling.
            blank (bool, optional): If True images will not be merged.

        TODO:
            Scaling.
        """
        debug('blitting surfaces', short=False)

        if ur_corner is None:
            try:
                x, y = ll_corner
                if not blank:
                    self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :] = \
                        AxisSurface.merge_images(self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :], surface.bitmap)
                else:
                    self.bitmap[x:x + surface.res[0], y:y + surface.res[1], :] = surface.bitmap
                    self.bitmap[:, :, 3] = 1
            except IndexError:
                raise IndexError("Given surface is too big.")

    def generate_png(self, filename):
        """
        Generates png out of bitmap.
        """
        debug('generating png', short=False)

        scaled_alpha = self.bitmap.astype('uint8')
        scaled_alpha = np.transpose(scaled_alpha, (1, 0, 2))[::-1, :, :]
        scaled_alpha[:, :, 3] *= 255
        img = Image.fromarray(scaled_alpha)
        img.save(filename)

    @staticmethod
    @deprecation.deprecated(deprecated_in='1.0.1', details='Too slow.')
    def slower_merge_images(bottom_img, top_img):
        """
        Puts img2 on top of img1.
        Args:
            bottom_img: First (bottom) image.
            top_img: Second (top) image.

        Returns:
            np.array: Merged image.

        Raises:
            ValueError: Image are not in the same shape.
        """
        debug('merging images', short=False)

        if bottom_img.shape != top_img.shape:
            raise ValueError

        result = np.zeros(bottom_img.shape)
        for x, y in itertools.product(range(bottom_img.shape[0]), range(bottom_img.shape[1])):
            alpha1 = bottom_img[x, y, 3]
            alpha2 = top_img[x, y, 3]
            for channel in range(3):
                result[x, y, channel] = alpha2 * top_img[x, y, channel] + alpha1 * (1 - alpha2) * bottom_img[x, y, channel]
            result[x, y, 3] = 1-(1-alpha1)*(1-alpha2)
        return result

    @staticmethod
    def merge_images(bottom_img, top_img):
        """
        Puts img2 on top of img1.
        Args:
            bottom_img: First (bottom) image.
            top_img: Second (top) image.

        Returns:
            np.array: Merged image.

        Raises:
            ValueError: Image are not in the same shape.
        """
        debug('merging images', short=False)

        if bottom_img.shape != top_img.shape:
            raise ValueError

        result = np.zeros(bottom_img.shape)
        alpha1 = np.array([top_img[:, :, 3], top_img[:, :, 3], top_img[:, :, 3]]).swapaxes(0, 2).swapaxes(0, 1)
        alpha2 = np.array([bottom_img[:, :, 3], bottom_img[:, :, 3], bottom_img[:, :, 3]]).swapaxes(0, 2).swapaxes(0, 1)
        result[:, :, :3] = alpha1 * top_img[:, :, :3] + alpha2 * (1 - alpha1) * bottom_img[:, :, :3]
        result[:, :, 3] = 1 - (1 - bottom_img[:, :, 3]) * (1 - top_img[:, :, 3])
        return result

    @staticmethod
    def merge_images_different_sized(bottom_img, top_img, corner):
        """
        Merging smaller or equal bitmap on top of the bigger one.

        Args:
            bottom_img (np.array): Bigger or equal, bottom image bitmap.
            top_img (np.array): Smaller or equal, top image bitmap.
            corner (tuple): Coordinates bottom-left corner of smaller image.
        """
        if bottom_img.shape[0] < top_img.shape[0] or bottom_img.shape[1] < top_img.shape[1]:
            raise ValueError('Bottom image should nod be smaller than the top one.')
        debug('merging images', short=True)
        x1, y1 = corner
        size_x, size_y, _ = top_img.shape
        x2, y2 = x1 + size_x, y1 + size_y
        result = copy.copy(bottom_img)
        alpha1 = np.array([top_img[:, :, 3]]*3).swapaxes(0, 2).swapaxes(0, 1)
        alpha2 = np.array([bottom_img[x1:x2, y1:y2, 3]]*3).swapaxes(0, 2).swapaxes(0, 1)
        result[x1:x2, y1:y2, :3] = alpha1 * top_img[:, :, :3] + alpha2 * (1 - alpha1) * bottom_img[x1:x2, y1:y2, :3]
        result[x1:x2, y1:y2, 3] = 1 - (1 - bottom_img[x1:x2, y1:y2, 3]) * (1 - top_img[:, :, 3])
        return result

    def check_if_point_is_valid(self, point):
        """
        Check if point in pixel coordinates is valid point on this surface.
        If abstract_coords is True, point is treated as in abstract coordinates.

        Args:
            point (tuple of ints): Coordinates of the point.

        Returns:
            bool: True if point is valid.
        """
        debug('checking if point is valid', short=True)
        x, y = point
        return 0 <= x < self.res[0] and 0 <= y < self.res[1]

    def blit_distinct_bitmap_objects(self, centers, img_objects, settings):
        """
        Blitting bitmap objects to the surface if it fits.
        When objects are not distinct, their alpha channel wil be ignored.

        Args:
            centers (list or tuple): List of center coords in pixel coordinates.
            img_objects (list or objects.BitmapObject): List of BitmapObjects to be blitted.
            settings (dict): Blitting settings.
        """

        blur = 0 if settings is None or 'blur' not in settings.keys() else settings['blur']
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']

        debug('blitting bitmap object', short=False)
        tmp_bitmap = np.zeros(self.res + (4,))

        for img_object, center in zip(img_objects, centers):
            x, y = np.array(center) - np.array(img_object.res) // 2
            if not self.check_if_point_is_valid((x, y)) or not \
                    self.check_if_point_is_valid(np.array(center)
                                                 + np.array(img_object.res) // 2):
                continue

            tmp_bitmap[x:x + img_object.res[0], y:y + img_object.res[1], :] = img_object.bitmap

        if blur_kernel == 'box' and blur > 1:
            kernel = np.zeros((blur, blur))
            kernel.fill(1/blur**2)

            # TODO: Other kernels
            tmp_bitmap[:, :, 3] = scipy.signal.convolve2d(tmp_bitmap[:, :, 3], kernel, mode='same')

        self.bitmap = self.merge_images(self.bitmap, tmp_bitmap)


class AxisSurface(Surface):
    """
    Surface representing R^2 plane.

    Attributes:
        res (tuple of ints):  Frame resolution in pixels.
            zero_coords (tuple of ints): Pixel coordinates for (0, 0) point
            x_bounds (tuple of ints): Interval of x axis to be shown
            y_bounds (tuple of ints): Interval of y axis to be shown
    """

    def __init__(self, res, x_bounds, y_bounds):
        """
        Args:
            res (tuple of ints):  Frame resolution in pixels.
            x_bounds (tuple of ints): Interval of x axis to be shown
            y_bounds (tuple of ints): Interval of y axis to be shown
        """

        super().__init__(res)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        # -(real zero point)*(real_spread)/(abstract spread)
        self.zero_coords = (-x_bounds[0]*self.res[0]/(x_bounds[1]-x_bounds[0]),
                            -y_bounds[0]*self.res[1]/(y_bounds[1]-y_bounds[0]))
        self.parametric_blitting_queue = []
        self.filled_blitting_queue = []
        self.parametric_queue_settings = None
        self.filled_queue_settings = None

    def transform_to_pixel_coordinates(self, point):
        """
        Returns pixel coordinates of the point in abstract coordinates.

        Args:
            point (tuple of ints): Point to be transformed.

        Returns:
            tuple of ints: Pixel coordinates of the point.
        """
        debug('transforming coordinates', short=True)
        x_res, y_res = self.res
        x_lower_bound, x_upper_bound = self.x_bounds
        y_lower_bound, y_upper_bound = self.y_bounds

        transformation_matrix = np.asarray([[x_res/(x_upper_bound-x_lower_bound), 0],
                                            [0, y_res/(y_upper_bound-y_lower_bound)]])

        # Affine transformation
        return tuple(map(round, np.array(point) @ transformation_matrix + np.array(self.zero_coords)))

    @staticmethod
    def static_transform_to_pixel_coordinates(point, x_bounds, y_bounds, resolution):
        """
        Returns pixel coordinates of the point in abstract coordinates using given bounds

        Args:
            point (tuple of ints): Point to be transformed.

        Returns:
            tuple of ints: Pixel coordinates of the point.
        """
        debug('transforming coordinates', short=True)
        x_res, y_res = resolution
        x_lower_bound, x_upper_bound = x_bounds
        y_lower_bound, y_upper_bound = y_bounds

        transformation_matrix = np.asarray([[x_res/(x_upper_bound-x_lower_bound), 0],
                                            [0, y_res/(y_upper_bound-y_lower_bound)]])

        zero = (-x_bounds[0]*resolution[0]/(x_bounds[1]-x_bounds[0]),
                -y_bounds[0]*resolution[1]/(y_bounds[1]-y_bounds[0]))

        # Affine transformation
        return tuple(map(round, np.array(point) @ transformation_matrix + np.array(zero)))

    def check_if_point_is_valid(self, point, abstract_coords=False):
        """
        Check if point in pixel coordinates is valid point on this surface.
        If abstract_coords is True, point is treated as in abstract coordinates.

        Args:
            point (tuple of ints): Coordinates of the point.
            abstract_coords (optional): Specify in which coordinates the point is written.

        Returns:
            bool: True if point is valid.
        """
        debug('checking if point is valid', short=True)
        if abstract_coords:
            point = self.transform_to_pixel_coordinates(point)
        x, y = point
        return 0 <= x < self.res[0] and 0 <= y < self.res[1]

    @staticmethod
    def _visual_enhancement(image, thickness, blur, blur_kernel, color):
        """
        Adding thickness and blur to the image.

        Args:
            image: Image to be processed
            thickness: Target thickness of the curve.
            blur: Blur size.
            blur_kernel: Blur type.
            color: Curve color in RGBA.

        Returns:
            np.array: Processed image.
        """
        debug('visual enhancement', short=True)

        color = objects.ColorParser.parse_color(color)
        target_image = np.zeros(image.shape + (4,))
        # if thickness != 1:
        for x, y in itertools.product(range(image.shape[0]), range(image.shape[1])):
            if image[x, y] == 1:
                for xi, yi in itertools.product(range(x-thickness, x+thickness+1), range(y-thickness, y+thickness+1)):
                    try:
                        if (x-xi)**2 + (y-yi)**2 <= thickness and xi >= 0 and yi >= 0:
                            target_image[xi, yi, :] = np.asarray(color)
                    except IndexError:
                        pass

        target_image[:, :, 0].fill(color[0])
        target_image[:, :, 1].fill(color[1])
        target_image[:, :, 2].fill(color[2])

        kernel = np.array([[1]])
        if blur_kernel == 'box' and blur != 0:
            kernel = np.zeros((blur, blur))
            kernel.fill(1/blur**2)

        # TODO: Other kernels
        if blur != 0:
            target_image[:, :, 3] = scipy.signal.convolve2d(target_image[:, :, 3], kernel, mode='same')

        return target_image

    def blit_parametric_object(self, obj, settings=None, interval_of_param=None, queue=False):
        """
        Blitting ParametricObject to the surface.

        Args:
            obj (objects.ParametricObject): Object to be blitted.
            settings (dict, optional): List of visual settings.
                Available keys:
                    * 'thickness' (int): thickness of the curve.
                    * 'blur' (int): Blur strength.
                    * 'blur kernel' (str): Kernel of the blur. Default is 'box'.
                        Possible values:
                            - 'box'
                            - 'gaussian'
                    * 'sampling rate' (int): Sampling rate. Default is 1.
                    * 'color' (tuple of ints): Color in RGBA
            interval_of_param (tuple of numbers, optional): First and last value of parameter to be shown.
                If not specified, the surfaces x_bound will be used
            queue (bool, optional): If True object will be added to blitting queue to be blitted later
        """
        if obj.bounds is not None and interval_of_param is None:
            interval_of_param = obj.bounds
        elif interval_of_param is None:
            interval_of_param = self.x_bounds

        if queue:
            if obj.bounds is None:
                obj.add_bounds(interval_of_param)
            self.parametric_blitting_queue.append(obj)
            self.parametric_queue_settings = settings
            return

        debug('blitting parametric object', short=True)
        tmp_bitmap = np.zeros(self.res)

        sampling_rate = 1 if settings is None or 'sampling rate' not in settings.keys() else settings['sampling rate']
        thickness = 1 if settings is None or 'thickness' not in settings.keys() else settings['thickness']
        blur = 0 if settings is None or 'blur' not in settings.keys() else settings['blur']
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']
        color = (0xFF, 0xFF, 0xFF, 1) if settings is None or 'color' not in settings.keys() else settings['color']

        for t in np.linspace(*interval_of_param, max(self.res)*sampling_rate):
            point = self.transform_to_pixel_coordinates(obj.get_point(t))

            if self.check_if_point_is_valid(point):
                tmp_bitmap[point] = 1

        processed_bitmap = self._visual_enhancement(tmp_bitmap, thickness, blur, blur_kernel, color)
        self.bitmap = self.merge_images(self.bitmap, processed_bitmap)

    def blit_parametric_queue(self):
        """
        Blit parametric objects from parametric queue together at once.
        """
        if not self.parametric_blitting_queue:
            return
        if self.res[0] <= 640:
            debug('bulletting parametric queue', short=False)
        else:
            debug('blitting parametric queue', short=False)
        for obj in self.parametric_blitting_queue:
            obj.shift_interval()
        obj = functools.reduce(lambda x, y: x.stack_parametric_objects(y), self.parametric_blitting_queue)
        self.blit_parametric_object(obj, self.parametric_queue_settings, obj.bounds)
        self.parametric_blitting_queue = []

    def blit_filled_object(self, filled_obj, settings, interval_of_param=None, queue=False):
        """
        Blitting filled object.

        Args:
            filled_obj (objects.FilledObject): Object to be blitted.
            settings (dict): Blitting settings.
            interval_of_param (tuple): Parameter boundaries.
            queue: If True object will be saved in queue instead of blitting instantly.
        """

        if interval_of_param is None:
            interval_of_param = self.x_bounds
        if filled_obj.interval is not None:
            interval_of_param = filled_obj.interval

        if queue:
            if filled_obj.interval is None:
                raise ValueError('interval is None')
                # filled_obj.add_interval(interval_of_param)
            self.filled_blitting_queue.append(filled_obj)
            self.filled_queue_settings = settings
            return

        debug('blitting filled object', short=True)

        function1 = filled_obj.function1
        function2 = filled_obj.function2

        tmp_bitmap = np.zeros(self.res)
        if filled_obj.is_rec():
            x0, x1 = interval_of_param
            x0, y0 = self.transform_to_pixel_coordinates(function1.get_point(x0))
            x1, y1 = self.transform_to_pixel_coordinates(function2.get_point(x1))
            y0, y1 = sorted([y0, y1])
            tmp_bitmap[x0:x1, y0:y1] = 1

        else:
            for t in np.linspace(*interval_of_param, max(self.res)*settings['sampling rate']):
                x, y1 = self.transform_to_pixel_coordinates(function1.get_point(t))
                x, y2 = self.transform_to_pixel_coordinates(function2.get_point(t))
                if self.check_if_point_is_valid((x, min(y1, y2))) and self.check_if_point_is_valid((x, max(y2, y1))):
                    tmp_bitmap[x, min(y1, y2):max(y2, y1) + 1] = 1
                else:
                    for y in range(min(y1, y2), max(y2, y1) + 1):
                        if self.check_if_point_is_valid((x, y)):
                            tmp_bitmap[x, y] = 1
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']
        processed_bitmap = self._visual_enhancement(tmp_bitmap, settings['thickness'], settings['blur'],
                                                    blur_kernel, settings['color'])
        self.bitmap = self.merge_images(self.bitmap, processed_bitmap)

    def blit_filled_queue(self):
        """
            Blit filled objects from filled queue together at once.
        """
        if not self.filled_blitting_queue:
            return
        for obj in self.filled_blitting_queue:
            obj.shift_interval()

        debug('blitting filled queue', short=False)

        if all(map(lambda x: x.is_rec(), self.filled_blitting_queue)):
            tmp_bitmap = np.zeros(self.res)
            for obj in self.filled_blitting_queue:
                x0, x1 = obj.interval
                x0, y0 = self.transform_to_pixel_coordinates(obj.function1.get_point(x0))
                x1, y1 = self.transform_to_pixel_coordinates(obj.function2.get_point(x1))
                y0, y1 = sorted([y0, y1])
                tmp_bitmap[x0:x1, y0:y1] = 1
            settings = self.filled_queue_settings
            blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']
            processed_bitmap = self._visual_enhancement(tmp_bitmap, settings['thickness'], settings['blur'],
                                                        blur_kernel, settings['color'])
            self.bitmap = self.merge_images(self.bitmap, processed_bitmap)
            self.filled_blitting_queue = []
            return

        obj = functools.reduce(lambda x, y: x.stack_filled_objects(y), self.filled_blitting_queue)
        self.blit_filled_object(obj, self.filled_queue_settings, obj.interval)
        self.filled_blitting_queue = []

    def blit_axes(self, settings, x_only=False):
        """
        Blitting axes to the surface.
        Args:
            settings (dict): Blitting settings
            x_only (bool, optional): Not adding y axis.
        """
        debug('blitting axes', short=False)

        x_axis = objects.ParametricObject(lambda x: x, lambda x: 0)
        y_axis = objects.ParametricObject(lambda x: 0, lambda x: x)
        self.blit_parametric_object(x_axis, settings, interval_of_param=self.x_bounds, queue=True)
        if not x_only:
            self.blit_parametric_object(y_axis, settings, interval_of_param=self.y_bounds)
        self.blit_parametric_queue()

    def blit_scale(self, settings, x_interval=None, x_length=None, y_interval=None, y_length=None):
        """
        Blitting scale to the axis.
        Args:
            settings (dict): Standard visual settings.
            x_interval (float, optional): Interval between points on x axis. If None grid on x axis will not be blitted.
            x_length (float, optional): Single line length on x axis.
            y_interval (float, optional): Interval between points on y axis. If None grid on y axis will not be blitted.
            y_length (float, optional): Single line length on y axis.
        """
        debug('blitting scale', short=False)

        def make_const(c):
            return lambda x: c

        if x_interval is not None:
            n = int((self.x_bounds[1] - self.x_bounds[0]) // x_interval) + 1
            graduation = np.linspace(start=x_interval*math.ceil(self.x_bounds[0]/x_interval),
                                     stop=x_interval*math.floor(self.x_bounds[1]/x_interval),
                                     num=n)

            lines = [objects.ParametricObject(make_const(point), lambda t: t, [-x_length, x_length])
                     for point in map(float, graduation)]
            grid = functools.reduce(lambda x, y: x.stack_parametric_objects(y), lines)
            self.blit_parametric_object(grid, settings, interval_of_param=(-x_length, (2*len(lines) - 1)*x_length))

        if y_interval is not None:
            n = int((self.y_bounds[1] - self.y_bounds[0]) // y_interval) + 1
            graduation = np.linspace(start=y_interval*math.ceil(self.y_bounds[0]/y_interval),
                                     stop=y_interval*math.floor(self.y_bounds[1]/y_interval),
                                     num=n)

            lines = [objects.ParametricObject(lambda t: t, make_const(point), [-y_length, y_length])
                     for point in map(float, graduation)]
            grid = functools.reduce(lambda x, y: x.stack_parametric_objects(y), lines)
            self.blit_parametric_object(grid, settings, interval_of_param=(-y_length, (2 * len(lines) - 1) * y_length))

    def blit_closed_point(self, coords, radius, settings, queue=False):
        """
        Blitting a disk.

        Args:
            coords (tuple): Center coords in abstract coordinates.
            radius (float): Radius in abstract coordinates
            settings (dict): Blitting settings.
            queue: If True object will be saved in queue instead of blitting instantly.
        """
        disk = objects.Disk(coords, radius)
        self.blit_filled_object(disk, settings, queue=queue)

    def blit_open_point(self, coords, radius, settings, queue=False):
        """
        Blitting a circle.

        Args:
            coords (tuple): Center coords in abstract coordinates.
            radius (float): Radius in abstract coordinates
            settings (dict): Blitting settings.
            queue: If True object will be saved in queue instead of blitting instantly.
        """
        circle = objects.Ellipse(coords, radius, radius)
        self.blit_parametric_object(circle, settings, circle.bounds, queue=queue)

    def blit_bitmap_object(self, center, img_object, settings, surface_coordinates=True):
        """
        Blitting single bitmap object, alias for blit_distinct_bitmap_objects with only one object in a list.
        """
        # self.blit_distinct_bitmap_objects([center], [img_object], settings, surface_coordinates=surface_coordinates)
        if surface_coordinates:
            center = self.transform_to_pixel_coordinates(center)
        blur = 0 if settings is None or 'blur' not in settings.keys() else settings['blur']
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']

        debug('blitting bitmap object', short=True)

        x, y = np.array(center) - np.array(img_object.res) // 2
        if not self.check_if_point_is_valid((x, y)) or not \
                self.check_if_point_is_valid(np.array(center)
                                             + np.array(img_object.res) // 2):
            return

            # tmp_bitmap[x:x + img_object.res[0], y:y + img_object.res[1], :] = img_object.bitmap

        if blur_kernel == 'box' and blur > 1:
            kernel = np.zeros((blur, blur))
            kernel.fill(1 / blur ** 2)

            # TODO: Other kernels
            blurred_object = copy.copy(img_object.bitmap)
            blurred_object[:, :, 3] = scipy.signal.convolve2d(img_object.bitmap[:, :, 3], kernel, mode='same')
        else:
            blurred_object = img_object.bitmap
        self.bitmap = self.merge_images_different_sized(self.bitmap, blurred_object, (x, y))

    def blit_distinct_bitmap_objects(self, centers, img_objects, settings, surface_coordinates=True):
        """
        Blitting bitmap objects to the surface if it fits.
        When objects are not distinct, their alpha channel wil be ignored.

        Args:
            centers (list or tuple): List of center coords in abstract coordinates.
            img_objects (list or objects.BitmapObject): List of BitmapObjects to be blitted.
            settings (dict): Blitting settings.
            surface_coordinates (bool): Type of coordinates.
        """

        if surface_coordinates:
            centers = list(map(lambda center: self.transform_to_pixel_coordinates(center), centers))
        super().blit_distinct_bitmap_objects(centers, img_objects, settings)

    def blit_closed_pixel_point(self, coords, radius, opacity, settings, padding=5):
        """
        Blitting a bitmap dot with parameters in pixel scale.

        Args:
            coords (tuple): Center coords in abstract coordinates.
            radius (int): Radius in pixels.
            opacity (float): Opacity. 0: fully transparent, 1: fully not transparent.
            settings (dict): Blitting settings.
            padding (tuple): Dot's padding in pixels.
        """
        disk = objects.BitmapDisk(radius, settings['color'], opacity, padding)
        self.blit_distinct_bitmap_objects(coords, disk, settings)

    def blit_open_pixel_point(self, coords, radius, opacity, settings, padding=5):
        """
        Blitting a bitmap circle with parameters in pixel scale.

        Args:
            coords (tuple): Center coords in abstract coordinates.
            radius (int): Radius in pixels.
            opacity (float): Opacity. 0: fully transparent, 1: fully not transparent.
            settings (dict): Blitting settings.
            padding (tuple): Dot's padding in pixels.
        """
        circle = objects.BitmapCircle(radius, settings['color'], settings['thickness'], opacity, padding)
        self.blit_distinct_bitmap_objects(coords, circle, settings)

    def blit_dashed_curve(self, obj, number_of_dashes, precision=50, settings=None,
                          interval_of_param=None, queue=False):
        """
        Blitting dashed curve given by parametric equations.

        Args:
            obj (ParametricObject): Curve represented by ParametricObject.
            number_of_dashes (int): Number of dashes to be constructed.
            precision (int, optional): Numerical precision of calculation of dashes position.
            settings (dict, optional): Blitting settings. Argument can be None only if queue is True.
            interval_of_param (tuple): Interval of curve parameter.
            queue (bool): Used for blitting several curves in a row with same settings. On the last curve queue must
                be equal to False and settings must be provided.
        """

        debug('calculating dashed curve', short=True)
        number_of_dashes = 2*number_of_dashes - 1

        if not queue and settings is None:
            raise ValueError('Settings not provided.')

        # m = max(self.res)*settings['sampling rate']

        if interval_of_param is None:
            interval_of_param = self.x_bounds

        # print(interval_of_param)

        def derivative(x, function, m):
            return (function(x + 1 / m) - function(x)) / (1 / m)

        def integral(function, interval, m):
            return sum([function(interval[0] + k * (interval[1] - interval[0]) / m) for k in range(m)]) * (
                        interval[1] - interval[0]) / m

        def arc_length(function_y, interval, m, function_x=lambda x: x):
            g = lambda x: math.sqrt((derivative(x, function_y, m)) ** 2 + (derivative(x, function_x, m)) ** 2)
            return integral(g, interval, m)

        piece_of_arc = arc_length(obj.y_function, interval_of_param, precision, obj.x_function) / number_of_dashes
        # print(f'piece of arc: {piece_of_arc}')
        partition = [interval_of_param[0]]
        for part_number in range(number_of_dashes):
            # print(f'now {part_number}')
            for j in range(1, precision + 1):
                if arc_length(obj.y_function,
                              (partition[part_number],
                               partition[part_number] + j * (interval_of_param[1]-interval_of_param[0]) / precision),
                              precision, obj.x_function) >= piece_of_arc:
                    # print(arc_length(function_y, (partition[i], partition[i]+j/m), m, function_x))
                    # print(partition[part_number]+j/m)
                    partition.append(partition[part_number] + j*(interval_of_param[1]-interval_of_param[0]) / precision)
                    break

        # more_sampling = [(partition[i], (partition[i + 1] + partition[i]) / 2) for i in range(len(partition) - 1)]
        for interval in zip(partition[::2], partition[1:][::2]):
            one_dash = copy.copy(obj)
            one_dash.add_bounds(interval)
            self.blit_parametric_object(one_dash, settings, interval_of_param=interval, queue=True)

        if not queue:
            self.blit_parametric_queue()

    def blit_3d_object(self, obj, settings, point_transformation=lambda x: x):
        points = []
        for u in np.linspace(*obj.u_bounds, settings['u sampling']):
            for v in np.linspace(*obj.v_bounds, settings['v sampling']):
                p = (obj.x_equation(u, v), obj.y_equation(u, v), obj.z_equation(u, v))
                if None not in p:
                    p = np.array(p)
                    points.append(point_transformation(p))

        points.sort(key=lambda x: x[2], reverse=True)
        min_z, max_z = points[-1][2], points[0][2]
        spread = max_z - min_z
        c1, c2 = settings['front color'], settings['back color']

        for point in points:
            color = np.array(objects.ColorParser.blend(c2, c1, (max_z - point[2])/spread))
            pixel_point = self.transform_to_pixel_coordinates(point[:2])
            if self.check_if_point_is_valid(pixel_point):
                bottom_alpha = self.bitmap[pixel_point[0], pixel_point[1], 3]
                top_alpha = color[3]
                self.bitmap[pixel_point[0], pixel_point[1], :3] = top_alpha*color[:3] + bottom_alpha*(1 - top_alpha)*self.bitmap[pixel_point[0], pixel_point[1], :3]
                self.bitmap[pixel_point[0], pixel_point[1], 3] = 1 - (1 - bottom_alpha)*(1 - top_alpha)

    def blit_vector(self, start, end, settings_line, settings_arrow=None, arrow_path='img//arrow.png',
                    position_correction=(0, 0), angle_correction=0):
        line = objects.PolygonalChain((start, end))
        arrow = objects.ImageObject(arrow_path)
        angle = math.atan((end[1] - start[1])/(end[0] - start[0]))
        arrow.rotate(angle + angle_correction + math.pi*int(end[1] < start[1]))
        arrow.fill_color(settings_line['color'])
        self.blit_parametric_object(line, settings=settings_line, interval_of_param=(0, .99))
        self.blit_bitmap_object(np.array(end) + np.array(position_correction), arrow, settings_arrow)


class Frame(Surface):
    """
    Special surface intended to represent one frame.
    """
    def __init__(self, res, bg_color, x_padding, y_padding):
        super().__init__(res)
        self.bitmap = np.zeros(res + (4,), dtype='uint8')
        for channel, color in enumerate(objects.ColorParser.parse_color(bg_color)):
            self.bitmap[:, :, channel].fill(color)
        self.x_padding = x_padding
        self.y_padding = y_padding


class OneAxisFrame(Frame):
    """
    Frame with only one axis.
    Class created to simplify most common type of frames, not offering anything new.
    """
    def __init__(self, res, bg_color='black', x_padding=100, y_padding=100):
        super().__init__(res, bg_color, x_padding, y_padding)
        self.axis_surface = None

    def add_axis_surface(self, x_bounds, y_bounds):
        """
        Setting single AxisSurface.
        Args:
            x_bounds (tuple): x axis interval in abstract coordinates.
            y_bounds (tuple): y axis interval in abstract coordinates.
        """
        self.axis_surface = AxisSurface((self.res[0]-2*self.x_padding, self.res[1]-2*self.y_padding),
                                        x_bounds, y_bounds)

    def blit_parametric_object(self, obj, settings):
        """
        Blit parametric object to the axis surface.
        Args:
            obj (objects.ParametricObject): Object to be blitted.
            settings (dict): Blitting settings.
        """
        self.axis_surface.blit_parametric_object(obj, settings)

    def blit_axis_surface(self, blank=True):
        """
        Blitting all queues and axis surface to the frame surface.
        """
        self.axis_surface.blit_parametric_queue()
        self.axis_surface.blit_filled_queue()
        self.blit_surface(self.axis_surface, (self.x_padding, self.y_padding), blank=True)

    def blit_x_grid(self, settings, interval, length):
        """
        Blitting grid on x axis of axis surface.

        Args:
            settings (dict): Blitting settings.
            interval (float): Grid interval in abstract coordinates.
            length (float): Single grid length in abstract coordinates.
        """
        self.axis_surface.blit_scale(settings, x_interval=interval, x_length=length)

    def blit_y_grid(self, settings, interval, length):
        """
        Blitting grid on y axis of axis surface.

        Args:
            settings (dict): Blitting settings.
            interval (float): Grid interval in abstract coordinates.
            length (float): Single grid length in abstract coordinates.
        """
        self.axis_surface.blit_scale(settings, y_interval=interval, y_length=length)

    def blit_axes(self, settings, x_only=False):
        """
        Blit axes image to the axis surface.
        Args:
            settings (dict or objects.ParametricBlittingSettings): Blitting settings.
            x_only (bool): If True only x axis will be blitted.
        """
        self.axis_surface.blit_axes(settings, x_only=x_only)


class ThreeAxisFrame(Frame):
    """
    Frame containing 3 independent AxisSurfaces, placed like:
        ******************************************************
        *  **************  ***************  **************** *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  *            *  *             *  *              * *
        *  **************  ***************  **************** *
        ******************************************************
    """
    def __init__(self, res, bg_color, x_padding, y_padding, left_mid_padding, right_mid_padding):
        """
        Args:
            res (tuple): Frame resolution.
            bg_color (str or tuple): Background color.
            x_padding (int): Right and left padding.
            y_padding (int): Top and bottom padding.
            left_mid_padding (int): Padding between left and middle AxisFrames.
            right_mid_padding (int): Padding between middle and right AxisFrames.
        """
        super().__init__(res, bg_color, x_padding, y_padding)
        self.axis_surface_left = None
        self.axis_surface_middle = None
        self.axis_surface_right = None
        self.left_mid_padding = left_mid_padding
        self.right_mid_padding = right_mid_padding

    def add_axis_surfaces(self, res_left, res_middle, res_right, bounds_left, bounds_middle, bounds_right):
        """
        Setting AxisSurfaces.
        Resolution of frames must fit the resolution of frame and paddings.

        Args:
            res_left (tuple): Resolution of left AxisFrame.
            res_middle (tuple): Resolution of middle AxisFrame.
            res_right (tuple): Resolution of right AxisFrame.
            bounds_left (tuple): x and y bounds of left AxisFrame.
            bounds_middle (tuple): x and y bounds of middle AxisFrame.
            bounds_right (tuple): x and y bounds of right AxisFrame.

        Raises:
            ValueError: If resolution don't fit frame resolution and paddings.
        """
        if 2*self.x_padding + self.right_mid_padding + self.left_mid_padding \
           + res_left[0] + res_right[0] + res_middle[0] != self.res[0]:
            raise ValueError("Given resolutions of frames don't fit on x axis.")
        if 2*self.y_padding + res_left[1] != self.res[1]:
            raise ValueError("Resolution of left frame doesn't fit on y axis.")
        if 2*self.y_padding + res_middle[1] != self.res[1]:
            raise ValueError("Resolution of middle frame doesn't fit on y axis.")
        if 2*self.y_padding + res_right[1] != self.res[1]:
            raise ValueError("Resolution of right frame doesn't fit on y axis.")

        self.axis_surface_left = AxisSurface(res_left, *bounds_left)
        self.axis_surface_middle = AxisSurface(res_middle, *bounds_middle)
        self.axis_surface_right = AxisSurface(res_right, *bounds_right)

    def add_equal_axis_surfaces(self, bounds_left, bounds_middle, bounds_right):
        """
        Setting AxisFrames of same resolution, calculated by frame resolution and paddings.
        """
        x = (self.res[0] - 2*self.x_padding - self.left_mid_padding - self.right_mid_padding)//3
        y = self.res[1] - 2*self.y_padding
        self.add_axis_surfaces((x, y), (x, y), (x, y), bounds_left, bounds_middle, bounds_right)

    def blit_axis_surfaces(self):
        """
        Blitting all queues in AxisFrames and AxisFrames to main Frame.
        """
        self.axis_surface_left.blit_parametric_queue()
        self.axis_surface_left.blit_filled_queue()
        self.axis_surface_right.blit_parametric_queue()
        self.axis_surface_right.blit_filled_queue()
        self.axis_surface_middle.blit_parametric_queue()
        self.axis_surface_middle.blit_filled_queue()
        self.blit_surface(self.axis_surface_left, (self.x_padding, self.y_padding))
        self.blit_surface(self.axis_surface_middle,
                          (self.x_padding + self.axis_surface_right.res[0] + self.left_mid_padding, self.y_padding))
        self.blit_surface(self.axis_surface_right,
                          (self.x_padding + self.axis_surface_right.res[0] + self.left_mid_padding
                           + self.axis_surface_middle.res[0] + self.right_mid_padding, self.y_padding))

    def blit_frames_around_axis_surfaces(self, settings, x_inner_bounds=0, y_inner_bounds=0):
        """
        Blitting boundaries of AxisSurfaces.

        Args:
            settings (dict): Blitting settings.
            x_inner_bounds (int, optional): Inner x padding in pixels.
            y_inner_bounds (int, optional): Inner y padding in pixels.
        """
        color = objects.ColorParser.parse_color(settings['color'])
        thickness = settings['thickness']

        def make_rec(res):
            x_size, y_size = res
            x_size += 2*x_inner_bounds
            y_size += 2*y_inner_bounds
            rec = np.zeros((x_size, y_size, 4))
            for i, c in enumerate(color):
                rec[:thickness+1, :, i] = c
                rec[x_size-thickness:, :, i] = c
                rec[:, :thickness+1, i] = c
                rec[:, y_size-thickness:, i] = c
            return objects.BitmapObject(rec, (x_size, y_size))

        centers = ((self.x_padding + self.axis_surface_left.res[0]//2,
                    self.y_padding + self.axis_surface_left.res[1]//2),
                   (self.x_padding + self.axis_surface_right.res[0] + self.left_mid_padding +
                    self.axis_surface_middle.res[0]//2, self.y_padding + self.axis_surface_middle.res[1]//2),
                   (self.x_padding + self.axis_surface_right.res[0] + self.left_mid_padding
                    + self.axis_surface_middle.res[0] + self.right_mid_padding + self.axis_surface_right.res[0]//2,
                    self.y_padding + self.axis_surface_right.res[1]//2))
        recs = [make_rec(self.axis_surface_left.res),
                make_rec(self.axis_surface_middle.res),
                make_rec(self.axis_surface_right.res)]
        self.blit_distinct_bitmap_objects(centers, recs, settings)


class Film:
    """
    Whole movie created out of frames.

    Attributes:
        fps (int): Frames per second.
        frames (list): List of frames.
        resolution (tuple of ints): Film resolution in pixels.
        frame_counter (int): Using for numbering frames in save_ram mode.
    """
    def __init__(self, fps, resolution, id=''):
        """
        Args:
            fps: Frames per second.
        """
        self.fps = fps
        self.frames = []
        self.resolution = resolution
        self.frame_counter = 0
        self.id = id

    def add_frame(self, frame, save_ram=False):
        """
        Adding one frame at the end of the frame list.

        Args:
            frame (Frame): Frame to be added.
            save_ram (bool): Save frames temporarily on hard disk.
        """
        if save_ram:
            try:
                os.mkdir('tmp')
            except FileExistsError:
                pass

            with open(f'tmp//f{self.id}_{self.frame_counter}.npy', 'wb') as file:
                np.save(file, frame.bitmap.astype("uint8"))
                debug('File saved', False)

            self.frame_counter += 1
            return
        self.frames.append(frame)

    def render(self, name='video.mp4', save_ram=False):
        """
        Render the movie.
        Args:
            name: Target file.
            save_ram: Read frames temporarily saved on hard disk.
        """
        debug('rendering', short=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, self.fps, self.resolution)

        if not save_ram:
            raw_frames = list(map(lambda x: np.swapaxes(x, 0, 1),
                              [f.bitmap.astype('uint8')[:, ::-1, :-1][:, :, ::-1] for f in self.frames]))

            # print(f'{len(raw_frames)}, {raw_frames[0].shape}')
            for f in raw_frames:
                out.write(f)

        else:
            for n in range(self.frame_counter):

                f = np.load(f'tmp//f{self.id}_{n}.npy').astype('uint8')[:, ::-1, :-1][:, :, ::-1]
                f = np.swapaxes(f, 0, 1)
                out.write(f)
            # shutil.rmtree('tmp')
        print(f'saved to {name}')
        out.release()


class MultiDifferentialAnimation:
    def __init__(self, frame_generator, *differentials):
        self.frame_generator = frame_generator
        self.differentials = differentials

    def render(self, filename, settings, save_ram=False, id_='', start_from=0, read_only=False,
               precision=1000, speed=1, integrated=False):
        """
        Rendering the animation.

        Args:
            filename (str): Name of file of the video to be saved in.
            settings (dict or object.RenderSettings): Render settings.
            save_ram (bool, optional): If True, frames will be saved in tmp directory and not stored in RAM memory.
            id_ (str, optional): Prefix of temporary files. Change it in case of rendering more than one video at once or
                needing to keep frames saved.
            start_from (int, optional): Number of starting frame.
            read_only (bool, optional): If True the frames will not be generated, only read from tmp directory. Use it
                in case of previously interrupted render or rendering not from the beginning.
            precision (int): Precision of numerical computation of differential.
            speed (float): Value of differential t per second.
        """
        duration = settings['duration']
        film = Film(settings['fps'], settings['resolution'], id=id_)
        fps = settings['fps'] // speed

        def integral(foo):
            return lambda h: sum([foo((k+1) / (fps * precision)) for k in range(math.floor(h * fps) * precision)]) / (
                                  fps * precision)

        if not integrated and not isinstance(self.differentials[0], objects.ElementalFunction):
            t = [integral(differential) for differential in self.differentials]
        elif not integrated:
            def make_integral(differential):
                return lambda h: differential.integral(0, h)
            t = [make_integral(differential) for differential in self.differentials]
        else:
            t = self.differentials

        prev_t = [math.nan for _ in self.differentials]
        last_frame = None
        for dt in np.arange(start_from / fps, duration, 1 / fps):
            debug(
                f'[{round(dt * fps)}/{round(fps * duration)} '
                f'({int(100 * (round(dt * fps)) / (fps * duration))}%)]',
                short=False)
            current_t = [dif(dt) for dif in t]
            if read_only:
                film.frame_counter += 1
            elif all(np.isclose(current_t, prev_t, atol=1 / (10 * precision))):
                film.add_frame(last_frame, save_ram=save_ram)
            else:
                last_frame = self.frame_generator(*current_t)
                film.add_frame(last_frame, save_ram=save_ram)
            prev_t = copy.copy(current_t)
        film.render(filename, save_ram)

    @staticmethod
    def blend_lists(lists, t):
        """
        Returning convex combination of vectors based on parameter t.

        Args:
            lists (list): List of vectors (python lists, tuples or np.arrays)
            t (float): parameter of convex combination. Floor of this value will index the pair of vectors,
                fraction part will be the combination parameter.
        """
        return list((1 - t + math.floor(t)) * np.array(lists[math.floor(t)])
                    + (t - math.floor(t)) * np.array(lists[min(math.floor(t) + 1, len(lists) - 1)]))

    @staticmethod
    def blend_functions(foos, t):
        """
        Returning convex combination of functions based on parameter t.

        Args:
            foos (list): List of functions.
            t (float): parameter of convex combination. Floor of this value will index the pair of vectors,
                fraction part will be the combination parameter.        """
        return lambda x: (1 - t + math.floor(t)) * foos[math.floor(t)](x) + \
                         (t - math.floor(t)) * foos[min(math.floor(t) + 1, len(foos) - 1)](x)


class SingleAnimation(MultiDifferentialAnimation):
    """
        Smoothly animate single variable change.

        Attributes:
            frame_generator (func): Function with argument t returning Frame.
            differential (func): Function R->R specifying growth rate per second.
        """
    def __init__(self, frame_generator, differential):
        super().__init__(frame_generator, differential)

    @deprecation.deprecated(deprecated_in='1.2.0')
    def _render(self, filename, settings, save_ram=True, id_='', start_from=0, read_only=False,
                precision=1000, speed=1):
        """
        Rendering the animation.

        Args:
            filename (str): Name of file of the video to be saved in.
            settings (dict): Render settings.
            save_ram (bool, optional): If True, frames will be saved in tmp directory and not stored in RAM memory.
            id_ (str, optional): Prefix of temporary files. Change it in case of rendering more than one video at once or
                needing to keep frames saved.
            start_from (int, optional): Number of starting frame.
            read_only (bool, optional): If True the frames will not be generated, only read from tmp directory. Use it
                in case of previously interrupted render or rendering not from the beginning.
            precision (int): Precision of numerical computation of differential.
            speed (float): Value of differential t per second.
        """
        duration = settings['duration']
        film = Film(settings['fps'], settings['resolution'], id=id_)
        fps = settings['fps']//speed

        t = lambda h: sum([self.differentials[0](k/(fps*precision)) for k in range(math.floor(h*fps)*precision)])/(fps*precision)
        prev_t = math.nan
        last_frame = None
        for dt in np.arange(start_from/fps, duration, 1/fps):
            debug(f'[{round(dt*fps)}/{round(fps*duration)} ({int(100*(round(dt*fps))/(fps*duration))}%), t={t(dt): .3f}]', short=False)
            if read_only:
                film.frame_counter += 1
            elif math.isclose(t(dt), prev_t, abs_tol=1/(10*precision)):
                film.add_frame(last_frame, save_ram=save_ram)
            else:
                last_frame = self.frame_generator(t(dt))
                film.add_frame(last_frame, save_ram=save_ram)
            prev_t = t(dt)
        film.render(filename, save_ram)


class FunctionSequenceAnimation(SingleAnimation):
    """
    Smooth animation of sequence of functions.
    """
    def __init__(self, sequence, differential, frame_generator_from_foo):
        frame_generator = lambda t: frame_generator_from_foo(self.blend_functions(sequence, t))
        super().__init__(frame_generator, normalize_function(make_periodic(differential)))


class IntervalSequenceAnimation(SingleAnimation):
    """
    Smooth animation of sequence of vectors.
    """
    def __init__(self, sequence, differential, frame_generator_from_interval):
        frame_generator = lambda t: frame_generator_from_interval(self.blend_lists(sequence, t))
        super().__init__(frame_generator, differential)


class PipeInstance:
    def __init__(self, settings, obj=None, queue=False, blitting_type=None, **kwargs):
        self.obj = obj
        self.settings = settings
        self.queue = queue
        self.blitting_type = blitting_type
        self.kwargs = kwargs

    def blit(self, frame):
        if isinstance(self.obj, objects.ParametricObject):
            if 'interval_of_param' not in self.kwargs.keys():
                self.kwargs['interval_of_param'] = self.obj.bounds
            frame.axis_surface.blit_parametric_object(self.obj, settings=self.settings, queue=self.queue,
                                                      interval_of_param=self.kwargs['interval_of_param'])
            return
        if isinstance(self.obj, objects.FilledObject):
            if 'interval_of_param' not in self.kwargs.keys():
                self.kwargs['interval_of_param'] = self.obj.interval
            frame.axis_surface.blit_filled_object(self.obj, settings=self.settings, queue=self.queue,
                                                  interval_of_param=self.kwargs['interval_of_param'])
            return
        if self.blitting_type == 'bitmap':
            if 'surface_coordinates' not in self.kwargs.keys():
                self.kwargs['surface_coordinates'] = True
            if hasattr(self.obj, '__getitem__'):
                frame.axis_surface.blit_distinct_bitmap_objects(self.kwargs['centers'], self.obj, self.settings,
                                                                surface_coordinates=self.kwargs['surface_coordinates'])
            else:
                frame.axis_surface.blit_bitmap_object(self.kwargs['center'], self.obj, self.settings,
                                                      surface_coordinates=self.kwargs['surface_coordinates'])
        if self.blitting_type == 'axis' or self.blitting_type == 'axes':
            frame.axis_surface.blit_axes(self.settings)
        if self.blitting_type == 'x axis':
            frame.axis_surface.blit_axes(self.settings, x_only=True)
        if self.blitting_type == 'x scale':
            frame.axis_surface.blit_scale(self.settings, x_interval=self.kwargs['x_interval'], x_length=self.kwargs['x_length'])
        if self.blitting_type == 'y scale':
            frame.axis_surface.blit_scale(self.settings, y_interval=self.kwargs['y_interval'], y_length=self.kwargs['y_length'])
        if self.blitting_type == 'closed point':
            frame.axis_surface.blit_closed_point(self.kwargs['coords'], self.kwargs['radius'], self.settings, self.queue)
        if self.blitting_type == 'open point':
            frame.axis_surface.blit_open_point(self.kwargs['coords'], self.kwargs['radius'], self.settings, self.queue)
        if self.blitting_type == 'dash':
            if 'precision' not in self.kwargs.keys():
                self.kwargs['precision'] = 50
            frame.axis_surface.blit_dashed_curve(self, self.obj, self.kwargs['number_of_dashes'],
                                                 precision=self.kwargs['precision'], settings=self.settings,
                                                 interval_of_param=None, queue=self.queue)
        if self.blitting_type == 'vector':
            if 'settings_arrow' not in self.kwargs.keys():
                self.kwargs['settings_arrow'] = None
            if 'arrow_path' not in self.kwargs.keys():
                self.kwargs['arrow_path'] = 'img//arrow.png'
            if 'position_correction' not in self.kwargs.keys():
                self.kwargs['position_correction'] = (0, 0)
            if 'angle_correction' not in self.kwargs.keys():
                self.kwargs['angle_correction'] = 0
            frame.axis_surface.blit_vector(self.kwargs['start'], self.kwargs['end'], self.settings,
                                           settings_arrow=self.kwargs['settings_arrow'],
                                           arrow_path=self.kwargs['arrow_path'],
                                           position_correction=self.kwargs['position_correction'],
                                           angle_correction=self.kwargs['angle_correction'])


class AnimationPipeline:
    def __init__(self, elements_generator, bounds_generator, differentials):
        self.elements_generator = elements_generator
        self.bounds_generator = bounds_generator
        self.differentials = differentials

    def render(self, filename, settings, png_filename=None, speed=1, id_='dupa', start_from=0, read_only=False,
               precision=1000, integrated=False):
        def frame_generator(*t):
            frame = OneAxisFrame(settings['resolution'], settings['bg color'], settings['x padding'], settings['y padding'])
            frame.add_axis_surface(*self.bounds_generator(*t))
            elements = self.elements_generator(*t)
            for instance in elements:
                instance.blit(frame)
            frame.blit_axis_surface()
            if png_filename:
                frame.generate_png(filename)
            return frame

        animator = MultiDifferentialAnimation(frame_generator, *self.differentials)
        animator.render(filename, settings, save_ram=True, speed=speed, id_=id_, start_from=start_from,
                        read_only=read_only, precision=precision, integrated=integrated)


class AnalyticGeometry:
    @staticmethod
    def rotation_matrix(angle):
        return objects.PredefinedSettings.rotation_matrix(angle)

    @staticmethod
    def rotate_point(point, angle):
        print(AnalyticGeometry.rotation_matrix(angle))
        return (AnalyticGeometry.rotation_matrix(angle) @ np.array(point)[:, np.newaxis]).T

    @staticmethod
    def crossing_point_of_functions(foo1, foo2, domain, precision=1000):
        f = objects.Function(lambda x: foo1(x) - foo2(x))
        x = f.zeros(domain, precision=precision)[0]
        return x, foo1(x)

    @staticmethod
    def line_between_points(p1, p2, interval):
        return objects.ParametricObject(lambda t: p1[0] + t * p2[0], lambda t: p1[1] + t * p2[1],
                                        bounds=interval)













if __name__ == '__main__':
    """
    
    Don't make complicated tests here - use testing.py file.
    
    """
    print('dupa')

