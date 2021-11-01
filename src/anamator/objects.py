import math
import warnings
import itertools
import copy

import numpy as np
import deprecation

from abc import ABC, abstractmethod
from PIL import Image
from scipy.special import erf
from math import pi, sin, cos, exp


HD = (1280, 720)
FHD = (1920, 1080)
SHIT = (640, 480)


class ColorParser:
    @staticmethod
    def parse_color(color):
        """
        Parsing color.

        Possible formats:
            * hex value in string in RGB.
            * hex value in string in RGBA.
            * known name in string.
            * tuple or RGB values.
            * tuple of RGBA values

        Returns:
            Tuple of RGBA values.
        """
        color_dict = {
            'black': (0, 0, 0, 1),
            'red': (255, 0, 0, 1),
            'green': (0, 255, 0, 1),
            'blue': (0, 0, 255, 1),
            'white': (255, 255, 255, 1),
            'gray': (128, 128, 128, 1),
            'light gray': (200, 200, 200, 1),
            'dark gray': (60, 60, 60, 1),
            'dark blue 1': (21, 76, 121, 1),
            'dark blue 2': (6, 57, 112, 1),
            'purple 1': (65, 64, 115, 1),
            'purple 2': (76, 57, 87, 1),
            'green 1': (121, 180, 115, 1),
            'green 2': (112, 163, 127, 1),
            'light red': (254, 147, 140, 1),
            'nice red 1': (254, 147, 140, 1),
            'tea green': (194, 234, 189, 1),
            'nice green 1': (194, 234, 189, 1),
            'melon': (252, 185, 178, 1),
            'red pastel 1': (252, 185, 178, 1),
            'apricot': (254, 208, 187, 1),
            'red pastel 2': (254, 208, 187, 1),
            'champagne pink': (255, 229, 217, 1),
            'indian red': (201, 93, 99, 1),
            'orange yellow crayola': (255, 214, 112, 1),
            'blue bell': (143, 149, 211, 1),
            'thistle': (211, 196, 227, 1),
            'pistachio': (211, 196, 227, 1),
            'nyanza': (228, 253, 225, 1),
            'columbia blue': (206, 229, 242, 1),
            'beau blue': (172, 203, 225, 1),
            'cerulean frost': (124, 152, 179, 1),
            'silver pink': (213, 176, 172, 1),
            'tuscany': (206, 160, 174, 1),
            'eggplant': (104, 69, 81, 1),
            'eton blue': (186, 198, 159, 1),
            'cadet gray': (153, 161, 166, 1),
            'beige': (237, 240, 218, 1),
            'irresistible': (170, 68, 101, 1),
            'grullo': (168, 155, 140, 1),
            'banana mania': (240, 223, 173, 1),
            'coyote brown': (143, 92, 56, 1),
            'vivid tangerine': (235, 148, 134, 1),
            'magic mint': (182, 239, 212, 1),
            'baby powder': (255, 252, 249, 1),
            'shimmering blush': (220, 117, 143, 1),
            'honeydew': (240, 255, 241, 1),
            'opal': (171, 200, 199, 1),
            'monatee': (171, 171, 189, 1),
            'baby blue eyes': (162, 210, 255, 1),
            'uranian blue': (189, 224, 254, 1),
            'blush': (234, 99, 140, 1),
            'claret': (137, 2, 62, 1),
            'independence': (52, 67, 94, 1),
            'pink': (248, 189, 196, 1)

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

    @staticmethod
    def parse_and_add_alpha(color, alpha):
        """
        Parsing color and adds to it given alpha value.

        Args:
            color (str or tuple): Color to be parsed.
            alpha (float): Alpha value ranging from 0 to 1.

        Returns:
            tuple: Color in RGBA.

        Raises:
            ValueError: If t is not ranging from 0 to 1.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha must range from 0 to 1.')

        c = ColorParser.parse_color(color)
        return c[:3] + (alpha,)

    @staticmethod
    def blend(c1, c2, t):
        """
        Parse and blend colors by convex combination.

        Args:
            c1 (str or tuple): First color.
            c2 (str or tuple): Second color.
            t (float): Parameter of combination from [0, 1].

        Returns:
            tuple: Blended color in RGBA.

        Raises:
            ValueError: If t is not ranging from 0 to 1.
        """
        if t < 0 or t > 1:
            raise ValueError('t must be from [0, 1].')

        return tuple(np.array(ColorParser.parse_color(c1))*(1-t) + np.array(ColorParser.parse_color(c2))*t)


def normalize_function(foo, interval=(0, 1), precision=100):
    """
    Normalizes function on given interval.

    Args:
        foo (function): Function to be normalized.
        interval (tuple): Interval of normalization.
        precision (int): Precision of numerical calculations.

    Returns:
        function: Normalized function.
    """
    start, end = interval
    norm = sum([foo(start + k/precision) for k in range(math.floor(precision*(end-start)))])/math.floor(precision*(end-start))
    # print(norm)
    return lambda x: foo(x)/norm


def make_periodic(foo, t=1):
    """
    Making function periodic.

    Args:
        foo (function): Function to be made periodic.
        t (float): Period.

    Returns:
        function: Periodic function with period t.
    """
    return lambda x: foo(x-x//t*t)


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

    def __init__(self, image, res=None):
        super().__init__()
        self.bitmap = image

    @property
    def res(self):
        return self.bitmap.shape[:2]

    @res.setter
    def res(self, value):
        value = np.array(value)
        if value != self.bitmap.shape[:2]:
            raise ValueError(f'Shapes do not match: {value} vs {self.bitmap.shape[:2]}')

    def reshape(self, new_size):
        """
        Changes the resolution.

        Args:
            new_size: target resolution
        """
        if 0 in new_size:
            self.bitmap = np.array([[]]*4)
            return
        img = Image.fromarray(self.bitmap, "RGBA")
        img = img.resize(new_size)
        self.bitmap = np.array(img)

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
        if 0 in new_size:
            return np.array([[]]*4)
        img = Image.fromarray(img_bitmap, "RGBA")
        img = img.resize(new_size)
        return np.array(img)

    def fill_color(self, color):
        color = ColorParser.parse_color(color)[:3]
        self.bitmap[:, :, :3] = color

    def point_is_valid(self, point):
        x, y = point
        x_res, y_res = self.bitmap.shape[:2]
        return 0 <= x < x_res and 0 <= y < y_res

    def rotate(self, angle, new_instance=False, convolution_size=1, convolution_kernel='box'):
        rotation_matrix = np.array([[cos(angle), -sin(angle)],
                                    [sin(angle), cos(angle)]])
        reverse_rotation = np.linalg.inv(rotation_matrix)

        new_len = max(int(self.bitmap.shape[0]*1.5), int(self.bitmap.shape[1]*1.5))
        new_bitmap_shape = [new_len, new_len, self.bitmap.shape[2]]
        new_bitmap = np.zeros(new_bitmap_shape)
        x_size, y_size = new_bitmap.shape[:2]
        original_x, original_y = self.bitmap.shape[:2]
        for x in range(x_size):
            for y in range(y_size):
                corresponding_pixel = np.array([x-x_size//2, y-y_size//2]) @ reverse_rotation + np.array([original_x//2, original_y//2])
                c_x, c_y = list(map(int, corresponding_pixel))
                if convolution_size != 0 and self.point_is_valid((c_x-convolution_size, c_y-convolution_size)) \
                   and self.point_is_valid((c_x+convolution_size, c_y+convolution_size)):
                    if convolution_kernel == 'box':
                        new_bitmap[x, y, :] = np.mean(self.bitmap[c_x-convolution_size:c_x+convolution_size+1,
                                                      c_y-convolution_size:c_y+convolution_size+1, :], axis=(0, 1))
                    else:
                        new_bitmap[x, y, :] = np.mean(self.bitmap[c_x - convolution_size:c_x + convolution_size + 1,
                                                      c_y - convolution_size:c_y + convolution_size + 1,
                                                      :]*convolution_kernel, axis=(0, 1))
                elif self.point_is_valid((c_x, c_y)):
                    new_bitmap[x, y, :] = self.bitmap[c_x, c_y, :]

        if new_instance:
            return BitmapObject(new_bitmap, new_bitmap.shape[:2])
        self.bitmap = new_bitmap

    def generate_png(self, filename):
        scaled_alpha = self.bitmap.astype('uint8')
        # scaled_alpha = np.transpose(scaled_alpha, (1, 0, 2))[::-1, :, :]
        scaled_alpha[:, :, 3] *= 255
        img = Image.fromarray(scaled_alpha)
        img.save(filename)


class ImageObject(BitmapObject):
    """
    Image object.
    """

    def __init__(self, image_path, resolution=None, scale=None):
        """
        Args:
            image_path (str): Path to the graphics.
            resolution (tuple of ints): Target surface resolution


        Raises:
            AttributeError: If given image is not in RGBA
        """
        # super().__init__()
        img = Image.open(image_path)
        image = np.array(img).swapaxes(0, 1)[:, ::-1, :]
        if image.shape[2] != 4:
            raise AttributeError("Image has to be in RGBA")

        if resolution is None:
            res = image.shape[:2][::-1]
        else:
            res = resolution

        if scale is not None:
            resolution = res = list(map(lambda x: int(x*scale), res))

        if 0 in resolution:
            super().__init__(np.array([]), (0, 0))
            return

        if image.shape[:2] != res:
            image = BitmapObject.static_reshape(image, resolution)

        image[..., 3] = image[..., 3]/255

        super().__init__(image, res)


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

    def shift_interval(self):
        """
        Shifts bounds to start from 0.
        """
        if self.bounds is None:
            return
        x_function_copy = copy.copy(self.x_function)
        y_function_copy = copy.copy(self.y_function)
        b0 = self.bounds[0]
        self.x_function = lambda t: x_function_copy(t+b0)
        self.y_function = lambda t: y_function_copy(t+b0)
        self.bounds = [0, self.bounds[1]-self.bounds[0]]


def create_const(c):
    return lambda x: c


class Function(ParametricObject):
    def __init__(self, function=None, const=None):
        """
        Args:
            function (func): Function to be represented.
            const (float, optional): If not None, function parameter will be ignored and constant function will be
                created out of value of const. Where creating constant function, this way is preferred as it
                optimize implementation of some methods operating on Function objects.
        """
        if const is None:
            super().__init__(lambda x: x, function)
        else:
            super().__init__(lambda x: x, create_const(const))
        self.const = const

    def __call__(self, *args, **kwargs):
        """
        Calculating the function value at given argument.
        """
        if self.const is not None:
            return self.const
        return self.y_function(*args, **kwargs)

    def sup(self, interval=(0, 1), precision=50):
        """
        Finding the supremum of a function on given interval.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            float: Numerical approximation of supremum on given interval.
        """
        if self.const is not None:
            return self.const
        return max([self(x) for x in np.linspace(*interval, precision)])

    def inf(self, interval=(0, 1), precision=50):
        """
        Finding the infimum of a function on given interval.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            float: Numerical approximation of infimum on given interval.
        """
        if self.const is not None:
            return self.const
        return min([self(x) for x in np.linspace(*interval, precision)])

    def zeros(self, interval=(0, 1), precision=100):
        """
        Finding approximations of function's zeros on given interval.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            list: List of zeros.
        """
        if self.const == 0:
            return list(np.linspace(*interval, precision))
        if self.const is not None:
            return []
        division = [(i, self(i)) for i in np.linspace(*interval, precision)]
        result = []
        for p1, p2 in zip(division[:-1], division[1:]):
            x1, y1 = p1
            x2, y2 = p2
            if y1 == 0:
                result.append(x1)
            if y2 == 0:
                result.append(x2)
            if y1*y2 < 0:
                result.append((x1 + x2)/2)
        return result

    def one_sign_intervals(self, interval=(0, 1), precision=100):
        """
        Returns list of intervals on which function does not change sign.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            list: List of intervals.
        """
        points = [interval[0]] + self.zeros(interval, precision) + [interval[1]]
        points = sorted(list(set(points)))
        return list(zip(points[:-1], points[1:]))

    def argmin(self, interval=(0, 1), precision=100):
        """
        Finding the argmin on given interval.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            float: Argmin on given interval.
        """
        return min([(x, self(x)) for x in np.linspace(*interval, precision)], key=lambda x: x[1])[0]

    def argmax(self, interval=(0, 1), precision=100):
        """
        Finding the argmax on given interval.

        Args:
            interval (tuple): Interval of search.
            precision (float): Precision of numerical computations.

        Returns:
            float: Argmax on given interval.
        """
        return max([(x, self(x)) for x in np.linspace(*interval, precision)], key=lambda x: x[1])[0]


class FilledObject(Object):
    def __init__(self, function1, function2, interval):
        """
        Object bounded by two functions that can be filled

        Args:
            function1 (Function): First boundary function.
            function2 (Function): Second boundary function.
        """
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.function1.add_bounds(interval)
        self.function2.add_bounds(interval)
        self.interval = interval
        if interval is None:
            raise ValueError('Interval is None')

    @deprecation.deprecated(deprecated_in='1.0.0', details='Interval should always be specified in constructor.')
    def add_interval(self, interval):
        if interval is None:
            raise ValueError('Interval is None')
        self.interval = interval
        self.function1.add_bounds(interval)
        self.function2.add_bounds(interval)

    def stack_filled_objects(self, other):
        """
        Gluing two FilledObjects together.

        Args:
            other (FilledObject): FilledObject to be glued.

        Returns:
            FilledObject: New, merged FilledObject.
        """
        if self.interval is None:
            warnings.warn('FilledObject without interval')

        self.function1.add_bounds(self.interval)
        self.function2.add_bounds(self.interval)
        new_foo1 = self.function1.stack_parametric_objects(other.function1)
        new_foo2 = self.function2.stack_parametric_objects(other.function2)
        return FilledObject(new_foo1, new_foo2, new_foo1.bounds)

    def shift_interval(self):
        """
        Shifts interval to start from 0.
        """
        if self.interval is None:
            return
        self.function1.shift_interval()
        self.function2.shift_interval()
        self.interval = [0, self.interval[1]-self.interval[0]]

    def is_rec(self):
        """
        Check if object is a rectangle. Some computations on rectangles such as blitting are much faster.
        """
        try:
            return self.function1.const is not None and self.function2.const is not None
        except AttributeError:
            return False


class Disk(FilledObject):
    """
    Representing a filled disk as FilledObject.
    """
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


class Ellipse(ParametricObject):
    """
    Representing an ellipse as ParametricObject.
    """
    def __init__(self, center, r1, r2):
        x0, y0 = center
        foo1 = lambda t: r1*math.cos(t) + x0
        foo2 = lambda t: r2*math.sin(t) + y0
        interval = [0, 2.1*math.pi]
        super().__init__(foo1, foo2, interval)


class BitmapCircle(BitmapObject):
    """
    Representing a circle as BitmapObject.
    """
    def __init__(self, radius, color, thickness, opacity, padding=5):
        shape = 2 * (radius + padding) + 1
        bitmap = np.zeros((shape, shape, 4))
        center_coord = radius + padding + 1
        color = ColorParser.parse_color(color)
        for x, y in itertools.product(range(shape), range(shape)):
            if (radius - thickness)**2 <= (x-center_coord)**2 + (y-center_coord)**2 < radius**2:
                bitmap[x, y, 0] = color[0]
                bitmap[x, y, 1] = color[1]
                bitmap[x, y, 2] = color[2]
                bitmap[x, y, 3] = opacity
        super().__init__(bitmap, (shape, shape))


class PolygonalChain(ParametricObject):
    """
    Representing a polygonal chain as parametric object.
    """
    def __init__(self, points):
        interval = (0, len(points)-1)

        def x_foo(t):
            t = min(t, interval[-1] - 1e-9)
            return (t-math.floor(t))*points[math.floor(t)][0] + (1-(t-math.floor(t)))*points[math.floor(t)+1][0]

        def y_foo(t):
            t = min(t, interval[-1] - 1e-9)
            return (t - math.floor(t)) * points[math.floor(t)][1] + (1 - (t - math.floor(t))) * points[math.floor(t) + 1][1]

        super().__init__(x_foo, y_foo, interval)


class LinearFunction(PolygonalChain):
    def __init__(self, a, b, bounds):
        foo = lambda x: a*x + b
        horizontal_0 = np.array((bounds[0][0], foo(bounds[0][0])))
        horizontal_1 = np.array((bounds[0][1], foo(bounds[0][1])))
        horizontal_vector = horizontal_1 - horizontal_0

        if a == 0:
            super().__init__(list(map(tuple, [horizontal_0, horizontal_1])))
            return

        vertical_0 = np.array(((bounds[1][0] - b)/a, foo((bounds[1][0] - b)/a)))
        vertical_1 = np.array(((bounds[1][1] - b)/a, foo((bounds[1][1] - b)/a)))
        vertical_vector = vertical_1 - vertical_0

        if np.linalg.norm(horizontal_vector) > np.linalg.norm(vertical_vector):
            super().__init__(list(map(tuple, [vertical_0, vertical_1])))
            return
        super().__init__(list(map(tuple, [horizontal_0, horizontal_1])))


class BitmapDisk(BitmapCircle):
    """
    Representing disk as BitmapObject.
    """
    def __init__(self, radius, color, opacity, padding=0):
        super().__init__(radius, color, radius, opacity, padding)


class Settings:
    def __init__(self, d=None, **kwargs):
        self.dictionary = d if d is not None else {}
        for key, value in kwargs:
            self.dictionary[key] = value

    def __getitem__(self, item):
        return self.dictionary[item]

    def keys(self):
        return self.dictionary.keys()


class RenderSettings(Settings):
    def __init__(self, fps=24, duration=1, resolution=FHD):
        dictionary = {
            'fps': fps,
            'duration': duration,
            'resolution': resolution
        }
        super().__init__(dictionary)


class ParametricBlittingSettings(Settings):
    def __init__(self, sampling_rate=3, thickness=5, color='white', blur=3, blur_kernel='box'):
        dictionary = {
            'sampling rate': sampling_rate,
            'thickness': thickness,
            'blur': blur,
            'color': color,
            'blur kernel': blur_kernel
        }
        super().__init__(dictionary)


class BitmapBlittingSettings(Settings):
    def __init__(self, blur=0, blur_kernel='box'):
        dictionary = {
            'blur': blur,
            'blur kernel': blur_kernel
        }
        super().__init__(dictionary)


class Parametric3DSettings(Settings):
    def __init__(self, u_sampling, v_sampling, f_color, b_color):
        dic = {
            'u sampling': u_sampling,
            'v sampling': v_sampling,
            'front color': ColorParser.parse_color(f_color),
            'back color': ColorParser.parse_color(b_color)
        }
        super().__init__(dic)


class PipelineSettings(Settings):
    def __init__(self, fps=24, duration=1, resolution=FHD, x_padding=100, y_padding=100, bg_color='black'):
        dictionary = {
            'fps': fps,
            'duration': duration,
            'resolution': resolution,
            'x padding': x_padding,
            'y padding': y_padding,
            'bg color': bg_color
        }
        super().__init__(dictionary)


@abstractmethod
class ElementalFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def integral(self, a, b):
        pass

    @abstractmethod
    def normalize(self, interval=(0, 1)):
        pass

    @abstractmethod
    def scale(self, factor):
        pass


class Gaussian(ElementalFunction):
    def __init__(self, center, exp_multiplier, normal_multiplier, normalization=(0, 1)):
        super().__init__()
        self.center = center
        self.exp_mul = exp_multiplier
        self.mul = normal_multiplier
        if normal_multiplier is None:
            self.mul = 1
            self.normalize(normalization)

    @property
    def foo(self):
        return lambda x: self.mul*math.exp(self.exp_mul*(x-self.center)**2)

    def __call__(self, *args, **kwargs):
        return self.foo(args[0])

    def integral(self, a, b):
        indefinite = lambda x: -math.pi**.5*self.mul*erf(-self.exp_mul**.5*(self.center - x))/(-2*self.exp_mul**.5)
        return indefinite(b) - indefinite(a)

    def derivative(self, x):
        return 2*self.mul*self.exp_mul*(x-self.center)*math.exp(self.exp_mul*(x-self.center)**2)

    def normalize(self, interval=(0, 1)):
        self.mul /= self.integral(*interval)

    def scale(self, factor):
        self.mul *= factor

    def __str__(self):
        return f'f(x) = {self.mul: .2f}exp({self.exp_mul: .2f}(x - {self.center:.2f}))'


class Polynomial(ElementalFunction):
    def __init__(self, coefs):
        super().__init__()
        self.coefs = coefs

    @property
    def foo(self):
        return lambda x: sum([c*x**i for i, c in enumerate(self.coefs)])

    def __call__(self, *args, **kwargs):
        return self.foo(args[0])

    def integral(self, a, b):
        new_coefs = [0] + [c/(i+1) for i, c in enumerate(self.coefs)]
        return Polynomial(new_coefs)(b) - Polynomial(new_coefs)(a)

    def derivative(self, x):
        new_coefs = [i*c for i, c in enumerate(self.coefs)][1:]
        return Polynomial(new_coefs)(x)

    def normalize(self, interval=(0, 1)):
        norm = self.integral(*interval)
        self.coefs = list(map(lambda x: x/norm, self.coefs))

    def scale(self, factor):
        self.coefs = list(map(lambda x: x*factor, self.coefs))

    def __str__(self):
        return ' + '.join([f'{c:.2f}x^{i}' for i, c in self.coefs])


class FunctionCutIntoParts(ElementalFunction):
    def __init__(self, breakpoints, elemental_functions):
        super().__init__()
        self.breakpoints = breakpoints
        self.elemental_functions = elemental_functions

    def __call__(self, *args, **kwargs):
        x = args[0]
        smaller_than_x = list(filter(lambda p: p[0] < x, zip(self.breakpoints, range(len(self.breakpoints)))))
        if not smaller_than_x:
            return self.elemental_functions[0](x)
        index = max(smaller_than_x, key=lambda p: p[0])[1] + 1
        return self.elemental_functions[index](x)

    def integral(self, a, b):
        value = 0

        for x0, x1, foo in zip(self.breakpoints[:-1], self.breakpoints[1:], self.elemental_functions):
            if x0 > b or x1 < a:
                continue
            value += foo.integral(max(x0, a), min(x1, b))

        return value

    def derivative(self, x):
        if x < self.breakpoints[0]:
            return self.elemental_functions[0].derivative(x)
        if x > self.breakpoints[-1]:
            return self.elemental_functions[-1].derivative(x)

        proper_foo = list(filter(lambda p: p[0] <= x < p[1], zip(self.breakpoints[:-1], self.breakpoints[1:],
                                                                self.elemental_functions)))[0][2]
        return proper_foo.derivative(x)

    def normalize(self, interval=(0, 1)):
        norm = self.integral(*interval)
        map(lambda x: x.scale(1/norm), self.elemental_functions)

    def scale(self, factor):
        map(lambda x: x.scale(factor), self.elemental_functions)


class PredefinedSettings:
    @staticmethod
    def fhd_render_24fps(duration):
        return {
            'fps': 24,
            'resolution': (1920, 1080),
            'duration': duration
        }

    @staticmethod
    def hd_render_24fps(duration):
        return {
            'fps': 24,
            'resolution': (1280, 720),
            'duration': duration
        }

    t5b2white = ParametricBlittingSettings(blur=2)
    hd_foo = t5b2white

    t5b0white = ParametricBlittingSettings(blur=0)
    fhd_axis = t5b0white

    t5b2gray = ParametricBlittingSettings(blur=2, color='gray')

    t0b0white = ParametricBlittingSettings(thickness=0, blur=0)
    white_filling = t0b0white

    t10b4white = ParametricBlittingSettings(thickness=10, blur=4)

    t10b3white = ParametricBlittingSettings(thickness=10)
    fhd_foo = t10b3white

    t10b4gray = ParametricBlittingSettings(thickness=10, blur=4, color='gray')

    t2b0white = ParametricBlittingSettings(thickness=2, blur=0)
    t2b0gray = ParametricBlittingSettings(thickness=2, blur=0, color='gray')
    t1b0white = ParametricBlittingSettings(thickness=1, blur=0)
    t1b0gray = ParametricBlittingSettings(thickness=1, blur=0, color='gray')

    slow_differential = make_periodic(normalize_function(lambda x: (x - 1/4) ** 2 * (3/4 - x) ** 2 if abs(x - 1 / 2) < 1/4 else 0))
    fast_differential = make_periodic(normalize_function(lambda x: (x - 3 / 8) ** 2 * (5 / 8 - x) ** 2 if abs(x - 1 / 2) < 1 / 8 else 0))
    exp_differential = make_periodic(normalize_function(lambda x: math.exp(-100*(x-1/2)**2)))

    @staticmethod
    def radius_func_creator(at_75=12, end=8):
        def radius(x):
            if x < .75:
                return int(at_75 / .75 * x)
            return int(4 * (end - at_75) * x - 3 * end + 4 * at_75)
        return radius

    @staticmethod
    def rotation_matrix(angle):
        return np.array([[cos(angle), -sin(angle)],
                         [sin(angle), cos(angle)]])


class Parametric3DObject:
    def __init__(self, equation1, equation2, equation3, bounds1, bounds2):
        self.x_equation = equation1
        self.y_equation = equation2
        self.z_equation = equation3
        self.u_bounds = bounds1
        self.v_bounds = bounds2



