import math
import warnings
import itertools

import numpy as np

from abc import ABC, abstractmethod
from PIL import Image


class ColorParser:
    @staticmethod
    def parse_color(color):
        color_dict = {
            'black': (0, 0, 0, 1),
            'red': (255, 0, 0, 1),
            'green': (0, 255, 0, 1),
            'blue': (0, 0, 255, 1),
            'white': (255, 255, 255, 1),
            'gray': (128, 128, 128, 1),
            'light gray': (200, 200, 200, 1),
            'dark gray': (60, 60, 60, 1)
        }
        if isinstance(color, str) and color[0] != '#':
            try:
                return color_dict[color]
            except IndexError:
                raise IndexError("Unknown color.")

        if isinstance(color, str) and len(color) == 7:
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16), 1

        if isinstance(color, str):
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), int(color[7:], 16)

        return color


class Object(ABC):
    """
    Abstract class for all objects
    """

    def __init__(self):
        super().__init__()


class BitmapObject(Object):
    """
    Objects with custom image.

    Attributes:
        image (np.array): custom image stored in numpy array (RGBA)
        res (tuple of ints): Image resolution
    """

    @abstractmethod
    def __init__(self, image, res):
        super().__init__()
        self.image = image
        self.res = res

    def reshape(self, new_size):
        """
        Changes the resolution.

        Args:
            new_size: target resolution
        """

        img = Image.fromarray(self.image, "RGBA")
        img.resize(new_size)
        self.image = np.array(img)
        self.res = new_size

    @staticmethod
    def static_reshape(img_bitmap, new_size):
        """
        Changes the resolution of given image

        Args:
            img_bitmap (np.array): Bitmap of image in numpy array.
            new_size: target resolution

        Returns:
            np.array: Reshaped image bitmap.
        """

        img = Image.fromarray(img_bitmap, "RGBA")
        img.resize(new_size)
        return np.array(img)


class Axis(BitmapObject):
    """
    Axis object.
    """

    def __init__(self, image_path, resolution=None):
        """
        Args:
            image_path (str): Path to the graphics.
            resolution (tuple of ints): Target surface resolution


        Raises:
            AttributeError: If given image is not in RGBA
        """
        super().__init__()
        img = Image.open(image_path)
        self.image = np.array(img)
        if self.image.shape[3] != 4:
            raise AttributeError("Image has to be in RGBA")

        if resolution is None:
            self.res = self.image.shape[:2]
        else:
            self.res = resolution

        if self.image.shape[:2] != self.res:
            self.image = BitmapObject.static_reshape(self.image, resolution)


class ParametricObject(Object):
    """
    Stores object defined by parametric equations.
    """

    def __init__(self, x_function, y_function, bounds=None):
        """
        Args:
            x_function (func): Function of x coordinates.
            y_function (func): Function of y coordinates.
        """
        super().__init__()
        self.x_function = x_function
        self.y_function = y_function
        self.bounds = bounds

    def get_point(self, t):
        """
        Returning the point at parameter value t.

        Args:
            t: Parameter value

        Returns:
            tuple of ints: Coordinates of calculated point.
        """
        return self.x_function(t), self.y_function(t)

    def add_bounds(self, bounds):
        self.bounds = bounds

    def stack_parametric_objects(self, other, t_threshold=None, inclusive=True):
        """
        Merging this parametric object with another.

        Example: if this parametric object represents a line, and another another line, then by stacking them
        we can obtain polygonal chain.

        Args:
            other (ParametricObject): Object to be merged with this one.
            t_threshold: The end value of t for the first object.
            inclusive: If True, value at t_threshold will be calculated with respect to this object.

        Returns:
            ParametricObject: Merged object.

        """
        if self.bounds is None:
            warnings.warn('self.bounds is None')
        if other.bounds is None:
            warnings.warn('other.bounds is None')

        spread = t_threshold
        if self.bounds is not None and t_threshold is None:
            t_threshold = self.bounds[1]
            spread = self.bounds[1] - self.bounds[0]

        x_function = lambda t: self.x_function(t) * int(t < t_threshold) + \
                               int(t >= t_threshold) * other.x_function(t - spread)
        y_function = lambda t: self.y_function(t) * int(t < t_threshold) + \
                               int(t >= t_threshold) * other.y_function(t - spread)
        if inclusive:
            x_function = lambda t: self.x_function(t) * int(t <= t_threshold) + \
                                   int(t > t_threshold) * other.x_function(t - spread)
            y_function = lambda t: self.y_function(t) * int(t <= t_threshold) + \
                                   int(t > t_threshold) * other.y_function(t - spread)

        if self.bounds is not None and other.bounds is not None:
            return ParametricObject(x_function, y_function,
                                    [self.bounds[0], self.bounds[1] + other.bounds[1] - other.bounds[0]])
        return ParametricObject(x_function, y_function)


class Function(ParametricObject):
    def __init__(self, function):
        """
        Args:
            function (func): Function to be represented.
        """
        super().__init__(lambda x: x, function)

    def __call__(self, *args, **kwargs):
        return self.y_function(*args, **kwargs)


class FilledObject(Object):
    def __init__(self, function1, function2, interval):
        """
        Object bounded by two functions that can be filled

        Args:
            function1: First boundary function.
            function2: Second boundary function.
        """
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.interval = interval
        if interval is None:
            raise ValueError('Interval is None')

    def add_interval(self, interval):
        if interval is None:
            raise ValueError('Interval is None')
        self.interval = interval

    def stack_filled_objects(self, other):
        if self.interval is None:
            warnings.warn('FilledObject without interval')

        self.function1.add_bounds(self.interval)
        self.function2.add_bounds(self.interval)
        new_foo1 = self.function1.stack_parametric_objects(other.function1)
        new_foo2 = self.function2.stack_parametric_objects(other.function2)
        return FilledObject(new_foo1, new_foo2, new_foo1.bounds)


class Disk(FilledObject):
    def __init__(self, center, radius):
        x0, y0 = center
        foo1 = Function(lambda x: abs(radius**2 - (x-x0)**2)**(1/2) + y0)
        foo2 = Function(lambda x: -(abs(radius ** 2 - (x - x0) ** 2) ** (1 / 2)) + y0)
        interval = [x0 - radius, x0 + radius]
        foo1.add_bounds(interval)
        foo2.add_bounds(interval)
        if interval is None:
            raise ValueError
        super().__init__(foo1, foo2, interval)


class Circle(ParametricObject):
    def __init__(self, center, radius):
        x0, y0 = center
        foo1 = lambda t: radius*math.cos(t) + x0
        foo2 = lambda t: radius*math.sin(t) + y0
        interval = [0, 2.1*math.pi]
        super().__init__(foo1, foo2, interval)


class BitmapCircle(BitmapObject):
    def __init__(self, radius, color, thickness, opacity, padding=5):
        shape = 2 * (radius + padding) + 1
        bitmap = np.zeros((shape, shape, 4))
        center_coord = radius + padding + 1
        color = ColorParser.parse_color(color)
        for x, y in itertools.product(range(shape), range(shape)):
            if (radius - thickness)**2 <= (x-center_coord)**2 + (y-center_coord)**2 < radius**2:
                bitmap[:, :, 0].fill(color[0])
                bitmap[:, :, 1].fill(color[1])
                bitmap[:, :, 2].fill(color[2])
                bitmap[:, :, 3].fill(opacity)
        super().__init__(bitmap, (shape, shape))


class BitmapDisk(BitmapCircle):
    def __init__(self, radius, color, opacity, padding=5):
        super().__init__(radius, color, 0, opacity, padding)


