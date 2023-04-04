import numpy as np
import matplotlib.pyplot as plt
import math as maths


# An array of points representing a straight line.
LINE = np.array([[0, 0], [1, 0]])

# An array of points representing the 2nd order koch curve. _/\_ <- But like, pointier.
ORDER_2_KOCH_CURVE = np.array(
    [[0, 0], [1, 0], [1.5, np.sqrt(3) / 2], [2, 0], [3, 0]])

# A reflection matrix that flips points along the x axis
FLIP_X_AXIS = np.array([[1, 0], [0, -1]])
# A reflection matrix that flips points along across the Y axis
FLIP_Y_AXIS = np.array([[-1, 0], [0, 1]])


def rotate(p: np.ndarray, origin: np.ndarray = (0, 0), degrees: float = 0):
    """
    Rotate a point or set of points by a specified number of degrees around a given origin.

    Args:
        p (array_like): The point or set of points to rotate. Must be a 2D array with shape (2,) or (n, 2).
        origin (array_like, optional): The origin around which to rotate the point(s). Defaults to (0, 0).
        degrees (float, optional): The angle in degrees by which to rotate the point(s). Defaults to 0.

    Returns:
        ndarray: The rotated point or set of points. Has the same shape as the input.

    Raises:
        ValueError: If p is not a 2D array with shape (2,) or (n, 2), or if origin is not a 2-tuple or 2-element array.

    Notes:
        This method was adapted from the solution provided by ImportanceOfBeingErnest on Stack Overflow:
        https://stackoverflow.com/a/58781388
    """
    angle = np.radians(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def kochCurve(order):

    if order == 1:
        return LINE

    kochCurve = ORDER_2_KOCH_CURVE

    for _ in range(0, order - 2):
        baseCurve = kochCurve.copy()
        curve = baseCurve.copy()

        dist = maths.dist(kochCurve[0], kochCurve[-1])
        curve[:, 0] += dist

        rotated = rotate(curve, curve[0], 60)
        curve = np.concatenate((baseCurve, rotated))

        lastPoint = curve[-1]
        curve[:, 0] -= lastPoint[0]
        mirrored = np.dot(curve.copy(), FLIP_Y_AXIS)
        curve[:, 0] += lastPoint[0]

        kochCurve = np.concatenate((curve, mirrored[::-1])) / 3

    return kochCurve


def arrange(curve):

    left = rotate(curve, curve[0], degrees=60)
    right = rotate(curve, curve[-1], degrees=-60)

    bottom = curve.copy().dot(FLIP_X_AXIS)

    return np.concatenate((bottom[::-1], left, right))


line = kochCurve(10)

line = arrange(line)

plt.figure(figsize=(5, 5))
plt.axis('equal')

plt.plot(line[:, 0], line[:, 1])
plt.show()
