import copy

import numpy as np

from abc import ABC, abstractmethod
from PIL import Image


class Object(ABC):
    """
    Abstract class for all objects
    """

    def __init__(self):
        super().__init__()


class ImageObject(Object):
    """
    Objects with custom image.

    Attributes:
        image (np.array): custom image stored in numpy array (RGBA)
        res (tuple of ints): Image resolution
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.image = None
        self.res = None

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


class Axis(ImageObject):
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
            self.image = ImageObject.static_reshape(self.image, resolution)


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
            print(f'new bounds: {[self.bounds[0], self.bounds[1] + other.bounds[1] - other.bounds[0]]}; '
                  f'\n first component starting value ({self.bounds[0]}): {(x_function(self.bounds[0]), y_function(self.bounds[0]))}'
                  f'\n second component ending value ({self.bounds[1] + other.bounds[1] - other.bounds[0]}): {(x_function(self.bounds[1] + other.bounds[1] - other.bounds[0]), y_function(self.bounds[1] + other.bounds[1] - other.bounds[0]))}')
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
