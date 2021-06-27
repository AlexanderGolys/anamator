import numpy as np

from PIL import Image
from abc import ABC


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
        self.res = res
        self.bitmap = np.zeros(self.res + (4,))

    def blit_surface(self, surface, ul_corner, lr_corner=None):
        """
        Merge surfaces. It scales the surface if lower right corner is provided.

        Args:
            surface (Surface): Surface to be blitted.
            ul_corner (tuple of ints): Upper left corner in pixel coordinates.
            lr_corner (tuple of ints, optional): Upper left corner in pixel coordinates. If provided surface will
                be scaled to fill given box. If None, surface wil be blitted without scaling.
        """


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

    def transform_to_surface_coordinates(self, point):
        """
        Returns pixel coordinates of the point in abstract coordinates.

        Args:
            point (tuple of ints): Point to be transformed.

        Returns:
            tuple of ints: Pixel coordinates of the point.
        """
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
        if abstract_coords:
            point = self.transform_to_surface_coordinates(point)
        x, y = point
        return 0 <= x < self.res[0] and 0 <= y < self.res[1]


class Object(ABC):
    """
    Abstract class for all objects
    """
    def __init__(self):
        super().__init__()

