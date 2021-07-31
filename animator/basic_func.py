import itertools
import functools
import math
import os

import numpy as np
import scipy.signal
import cv2

from PIL import Image

from animator import objects


DEBUG = True
DEBUG_SHORT = False


def debug(log, short=True):
    if DEBUG_SHORT or (DEBUG and not short):
        print(f'--- {log} ---')


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

    def blit_surface(self, surface, ll_corner, ur_corner=None):
        """
        Merge surfaces. It scales the surface if lower right corner is provided.

        Args:
            surface (Surface): Surface to be blitted.
            ll_corner (tuple of ints): Lower left corner in pixel coordinates.
            ur_corner (tuple of ints, optional): Upper right corner in pixel coordinates. If provided surface will
                be scaled to fill given box. If None, surface wil be blitted without scaling.

        TODO:
            Scaling.
        """
        debug('blitting surfaces', short=False)

        if ur_corner is None:
            try:
                x, y = ll_corner
                self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :] = \
                    AxisSurface.merge_images(self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :], surface.bitmap)
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

    def transform_to_surface_coordinates(self, point):
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
            point = self.transform_to_surface_coordinates(point)
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
        if thickness != 1:
            for x, y in itertools.product(range(image.shape[0]), range(image.shape[1])):
                if image[x, y] == 1:
                    for xi, yi in itertools.product(range(x-thickness, x+thickness+1), range(y-thickness, y+thickness+1)):
                        try:
                            if (x-xi)**2 + (y-yi)**2 <= thickness:
                                target_image[xi, yi, :] = np.asarray(color)
                        except IndexError:
                            pass

        target_image[:, :, 0].fill(color[0])
        target_image[:, :, 1].fill(color[1])
        target_image[:, :, 2].fill(color[2])

        kernel = np.array([[1]])
        if blur_kernel == 'box':
            kernel = np.zeros((blur, blur))
            kernel.fill(1/blur**2)

        # TODO: Other kernels
        if blur != 0:
            target_image[:, :, 3] = scipy.signal.convolve2d(target_image[:, :, 3], kernel, mode='same')

        return target_image

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
        for x, y in itertools.product(range(bottom_img.shape[0]), range(bottom_img.shape[1])):
            alpha1 = bottom_img[x, y, 3]
            alpha2 = top_img[x, y, 3]
            for channel in range(3):
                result[x, y, channel] = alpha2 * top_img[x, y, channel] + alpha1 * (1 - alpha2) * bottom_img[x, y, channel]
            result[x, y, 3] = 1-(1-alpha1)*(1-alpha2)
        return result

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
        if interval_of_param is None:
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
            point = self.transform_to_surface_coordinates(obj.get_point(t))

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
        debug('blitting parametric queue', short=False)
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
                filled_obj.add_interval(interval_of_param)
            self.filled_blitting_queue.append(filled_obj)
            self.filled_queue_settings = settings
            return

        debug('blitting filled object', short=True)

        function1 = filled_obj.function1
        function2 = filled_obj.function2

        tmp_bitmap = np.zeros(self.res)

        for t in np.linspace(*interval_of_param, max(self.res)*settings['sampling rate']):
            x, y1 = self.transform_to_surface_coordinates(function1.get_point(t))
            x, y2 = self.transform_to_surface_coordinates(function2.get_point(t))
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
        debug('blitting filled queue', short=False)
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

    def blit_bitmap_object(self, center, img_object, settings):
        """
        Blitting single bitmap object, alias for blit_distinct_bitmap_objects with only one object in a list.
        """
        self.blit_distinct_bitmap_objects([center], [img_object], settings)

    def blit_distinct_bitmap_objects(self, centers, img_objects, settings):
        """
        Blitting bitmap objects to the surface if it fits.
        When objects are not distinct, their alpha channel wil be ignored.

        Args:
            centers (list or tuple): List of center coords in abstract coordinates.
            img_objects (list or objects.BitmapObject): List of BitmapObjects to be blitted.
            settings (dict): Blitting settings.
        """

        blur = 0 if settings is None or 'blur' not in settings.keys() else settings['blur']
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']

        debug('blitting bitmap object', short=False)
        tmp_bitmap = np.zeros(self.res + (4,))

        for img_object, center in zip(img_objects, centers):
            x, y = np.array(self.transform_to_surface_coordinates(center)) + np.array(img_object.res)//2
            if not self.check_if_point_is_valid((x, y)) or not \
                    self.check_if_point_is_valid(np.array(self.transform_to_surface_coordinates(center))
                                                 + np.array(img_object.res) // 2):
                continue

            tmp_bitmap[x:x+img_object.res[0], y:y+img_object.res[1], :] = img_object.bitmap

        if blur_kernel == 'box':
            kernel = np.zeros((blur, blur))
            kernel.fill(1/blur**2)

            # TODO: Other kernels
            if blur != 0:
                tmp_bitmap[:, :, 3] = scipy.signal.convolve2d(tmp_bitmap[:, :, 3], kernel, mode='same')

        self.bitmap = self.merge_images(self.bitmap, tmp_bitmap)

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
    def __init__(self, res, bg_color, x_padding, y_padding):
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

    def blit_axis_surface(self):
        """
        Blitting all queues and axis surface to the frame surface.
        """
        self.axis_surface.blit_parametric_queue()
        self.axis_surface.blit_filled_queue()
        self.blit_surface(self.axis_surface, (self.x_padding, self.y_padding))

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
            settings (dict): Blitting settings.
            x_only (bool): If True only x axis will be blitted.
        """
        self.axis_surface.blit_axes(settings, x_only=x_only)


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
                              [f.bitmap.astype('uint8')[:, ::-1, :-1] for f in self.frames]))

            # print(f'{len(raw_frames)}, {raw_frames[0].shape}')
            for f in raw_frames:
                out.write(f)

        else:
            for n in range(self.frame_counter):

                f = np.load(f'tmp//f{self.id}_{n}.npy').astype('uint8')[:, ::-1, :-1]
                f = np.swapaxes(f, 0, 1)
                out.write(f)
            # shutil.rmtree('tmp')
        print(f'saved to {name}')
        out.release()


class SingleAnimation:
    """
        Smoothly animate single variable change.

        Attributes:
            frame_generator (func): Function with argument t returning Frame.
            differential (func): Function R->R specifying growth rate per second.
        """
    def __init__(self, frame_generator, differential):
        self.frame_generator = frame_generator
        self.differential = differential

    def render(self, filename, settings, save_ram=False, id_='', start_from=0, read_only=False, precision=1000):
        fps = settings['fps']
        duration = settings['duration']

        film = Film(fps, settings['resolution'], id=id_)
        t = lambda h: sum([self.differential(k/(fps*precision)) for k in range(math.floor(h*fps)*precision)])/(fps*precision)
        for dt in np.arange(start_from/fps, duration, 1/fps):
            if read_only:
                film.frame_counter += 1
            else:
                film.add_frame(self.frame_generator(t(dt)), save_ram=save_ram)
                debug(f'[{round(dt*fps)+1}/{round(fps*duration)}]', short=False)
        film.render(filename, save_ram)


class FunctionSequenceAnimation(SingleAnimation):
    def __init__(self, sequence, differential, frame_generator_from_foo):
        frame_generator = lambda t: frame_generator_from_foo(lambda x: (1-t+math.floor(t))*sequence[math.floor(t)](x)
                                            + (t-math.floor(t))*sequence[min(math.floor(t)+1, len(sequence)-1)](x))

        super().__init__(frame_generator, normalize_function(make_periodic(differential)))


def normalize_function(foo, interval=(0, 1), precision=100):
    start, end = interval
    norm = sum([foo(start + k/precision) for k in range(math.floor(precision*(end-start)))])/math.floor(precision*(end-start))
    print(norm)
    return lambda x: foo(x)/norm


def make_periodic(foo, t=1):
    return lambda x: foo(x-x//t*t)


if __name__ == '__main__':
    """
    
    Don't make complicated tests here - use testing.py file.
    
    """
    print('dupa')

