import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

from asa.Bcorner import quantile


class ContourChecker:
    """
    A class to create a contour line from given X, Y, Z data and
    provides a method to check if new data points (x, y) are inside or outside
    the contour.
    """

    def __init__(self, X, Y, Z, contour_level):
        """
        Initializes the ContourChecker with the given data and contour level.

        Args:
            X: 2D array of x-coordinates (from `np.meshgrid` usually).
            Y: 2D array of y-coordinates (from `np.meshgrid` usually).
            Z: 2D array of function values at each (x, y) coordinate.
            contour_level: The value of the contour line to create.
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        self.contour_level = contour_level
        self.paths = self._create_contour_path()

    def _create_contour_path(self):
        """
        Creates a matplotlib Path object representing the contour line.
        """
        contour = plt.contour(self.X,
                              self.Y,
                              self.Z,
                              levels=[self.contour_level])
        if not contour.collections:
            print(
                "Warning: No contour found at this level.  All points will be considered outside."
            )
            return None  # Return None if no contour is found

        paths = contour.collections[0].get_paths()
        for p in paths:
            # if not closed
            if p.codes[-1] != mpath.Path.CLOSEPOLY:
                plt.close()
                raise ValueError("Contour path is not closed")
        plt.close()  # Close the figure to avoid displaying the contour plot
        return paths

    def is_inside(self, x, y):
        """
        Checks if a given set of points (x, y) are inside the contour line.

        Args:
            x: A numpy array of x-coordinates of the points.
            y: A numpy array of y-coordinates of the points.  Must be the same length as x.

        Returns:
            A numpy array of boolean values, where True indicates the corresponding
            point is inside the contour, and False indicates it is outside.
            Returns a numpy array of all False values if no contour was found
            during initialization.
        """
        if self.paths is None:
            return np.full(x.shape, False)  # All False if no contour

        points = np.column_stack((x, y))  # Create (x, y) pairs

        return np.any([p.contains_points(points) for p in self.paths], axis=0)


def find_contour_outliers(x, y, level, return_checker=True, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    Z, x_edges, y_edges = np.histogram2d(x.flatten(), y.flatten(), **kwargs)
    x_cen = (x_edges[:-1] + x_edges[1:]) / 2
    y_cen = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_cen, y_cen)

    level_value = quantile(Z.flatten(), 1 - level, weights=Z.flatten())[0]

    print("Contour level:", level_value)

    checker = ContourChecker(X, Y, Z.T, level_value)

    if return_checker:
        return checker.is_inside(x, y), checker

    return checker.is_inside(x, y)
