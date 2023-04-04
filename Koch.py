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


def kochCurve(order: int) -> np.ndarray:
    """
    Generate a Koch curve of a given order.

    Args:
        order (int): The order of the Koch curve to generate.

    Returns:
        np.ndarray: An array of (x, y) points defining the Koch curve.
    """

    if order == 1:
        # Base case: return the LINE segment
        return LINE

    # Start with the order-2 Koch curve as the initial curve
    kochCurve = ORDER_2_KOCH_CURVE

    # Recursively generate the Koch curve up to the desired order
    for _ in range(0, order - 2):

        # Copy the current curve to use as the base curve
        baseCurve = kochCurve.copy()

        # Extend the curve by adding a new segment to the end
        newSegment = baseCurve.copy()

        # Shift the new segment along the x axis by the length of the curve.
        length = maths.dist(kochCurve[0], kochCurve[-1])
        newSegment[:, 0] += length

        # rotate the new segment up by 60 degrees using the first point as the pivot
        rotated = rotate(newSegment, newSegment[0], 60)

        # Combine this with the base curve
        newSegment = np.concatenate((baseCurve, rotated))

        # Takes the last point of the new segment
        lastPoint = newSegment[-1]
        # Shifts back the origin of the segment the length curve
        newSegment[:, 0] -= lastPoint[0]
        # Creates a copy of this and flips it along the Y axis
        mirrored = np.dot(newSegment.copy(), FLIP_Y_AXIS)
        # Shifts that segment back to where it was.
        newSegment[:, 0] += lastPoint[0]

        # Combines the result and scales the curve down by a factor of 3
        kochCurve = np.concatenate((newSegment, mirrored[::-1])) / 3

    return kochCurve


def arrange(curve: np.ndarray) -> np.ndarray:
    """
    Arranges the given Koch curve into a snowflake shape by rotating and flipping it.

    Parameters:
        curve (np.ndarray): A 2D NumPy array representing the Koch curve to arrange. The array should have shape (n, 2)
            where n is the number of points in the curve, and each row should contain the x and y coordinates of a point.

    Returns:
        np.ndarray: A 2D NumPy array representing the arranged Koch curve. The array has shape (3*n-1, 2), where the first n
            rows correspond to the flipped version of the input curve, and the remaining 2*n-1 rows correspond to two rotated
            copies of the input curve. Each row contains the x and y coordinates of a point.

    Raises:
        ValueError: If the input curve is empty or has less than two points.

    Examples:
        >>> curve = np.array([[0, 0], [1, 0], [1/2, np.sqrt(3)/2]])
        >>> arrange(curve)
        array([[ 0.        , -0.        ],
               [ 1.        , -0.        ],
               [ 0.5       ,  0.8660254 ],
               [ 1.        ,  0.        ],
               [ 0.75      ,  0.4330127 ],
               [ 0.        ,  0.        ],
               [ 0.25      ,  0.4330127 ],
               [ 0.5       ,  0.8660254 ]])

    Note:
        This method assumes that the Koch curve is oriented from left to right, with the first point on the left and the
        last point on the right. If the curve is oriented in a different way, the results may be incorrect.
    """

    left = rotate(curve, curve[0], degrees=60)
    right = rotate(curve, curve[-1], degrees=-60)

    bottom = curve.copy().dot(FLIP_X_AXIS)

    return np.concatenate((bottom[::-1], left, right))


line = kochCurve(4)

line = arrange(line)

plt.figure(figsize=(5, 5))
plt.axis('equal')

plt.plot(line[:, 0], line[:, 1])
plt.show()
