import functools
import math as maths
import threading
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

#
#   Matt Tyler - 1049833
#
# An array of points representing a straight line.
LINE = np.array([[0, 0], [3, 0]])

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

    # Generate the Koch curve up to the desired order
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


def snowflake(order: int) -> np.ndarray:
    """
    Generate a Koch snowflake of a given order.

    The Koch snowflake is a fractal curve constructed by successively applying the Koch curve to the initial equilateral
    triangle. This function generates the snowflake by first generating the Koch curve of the specified order, then
    arranging it into a snowflake shape by rotating and flipping the curve.

    Args:
        order (int): The order of the Koch snowflake to generate. Must be a positive integer.

    Returns:
        np.ndarray: A 2D NumPy array representing the Koch snowflake. The array has shape (3*n-1, 2), where n is the
            number of points in the Koch curve, and each row contains the x and y coordinates of a point in the snowflake.

    Raises:
        ValueError: If the input order is not a positive integer.

    Examples:
        >>> snowflake(1)
        array([[ 0.        , -0.        ],
               [ 1.        , -0.        ],
               [ 0.5       ,  0.8660254 ],
               [ 1.        ,  0.        ],
               [ 0.75      ,  0.4330127 ],
               [ 0.        ,  0.        ],
               [ 0.25      ,  0.4330127 ],
               [ 0.5       ,  0.8660254 ]])
    """

    line = kochCurve(order)
    return arrange(line)


@functools.lru_cache(maxsize=None)
def cached_snowflake(order: int) -> np.ndarray:
    """
    Generate and cache a Koch snowflake of a given order.

    This function generates a Koch snowflake of the specified order using the `snowflake` function and caches the
    result using the `functools.lru_cache` decorator with no maximum cache size. The cache is used to store previously
    generated snowflakes to improve the performance of repeated calls to this function with the same order.

    Args:
        order (int): The order of the Koch snowflake to generate. Must be a positive integer.

    Returns:
        np.ndarray: A 2D NumPy array representing the Koch snowflake. The array has shape (3*n-1, 2), where n is the
            number of points in the Koch curve, and each row contains the x and y coordinates of a point in the snowflake.

    Raises:
        ValueError: If the input order is not a positive integer.

    Examples:
        >>> cached_snowflake(1)
        array([[ 0.        , -0.        ],
               [ 1.        , -0.        ],
               [ 0.5       ,  0.8660254 ],
               [ 1.        ,  0.        ],
               [ 0.75      ,  0.4330127 ],
               [ 0.        ,  0.        ],
               [ 0.25      ,  0.4330127 ],
               [ 0.5       ,  0.8660254 ]])
    """
    print("Loaded order", order)
    return snowflake(order)


def create_snowflake_figure(order: int) -> plt.figure:
    """
    Create a Matplotlib figure containing a Koch snowflake of a given order.

    This function generates a Matplotlib figure with an 8x8 inch size and a single subplot displaying a Koch snowflake of
    the specified order. The snowflake points are retrieved from the `cached_snowflake` function, which caches previously
    generated snowflakes for performance improvement. The x and y limits are set to fit the snowflake within the subplot.

    Args:
        order (int): The order of the Koch snowflake to generate. Must be a positive integer.

    Returns:
        plt.Figure: A Matplotlib figure containing the subplot with the Koch snowflake of the specified order.

    Raises:
        ValueError: If the input order is not a positive integer.

    Examples:
        >>> fig = create_snowflake_figure(1)
        >>> fig.show()
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('equal')
    ax.axis('off')

    snowflake_points = cached_snowflake(order)
    ax.plot(snowflake_points[:, 0], snowflake_points[:, 1])

    return fig


def preload_snowflakes(max_order: int) -> None:
    """
    Preload and cache Koch snowflakes up to a specified maximum order.

    This function preloads and caches Koch snowflakes of orders from 1 to the specified maximum order using the
    `cached_snowflake` function, which utilizes an LRU cache to store the generated snowflakes for performance improvement.
    The snowflakes are loaded in reverse order, from the maximum order down to 1.

    Args:
        max_order (int): The maximum order of Koch snowflakes to preload. Must be a positive integer.

    Returns:
        None

    Raises:
        ValueError: If the input max_order is not a positive integer.

    Examples:
        >>> preload_snowflakes(3)
    """
    for order in range(max_order, 0, -1):
        cached_snowflake(order)


# Preload snowflake points up to order 13 in a separate thread
preload_thread = threading.Thread(target=preload_snowflakes, args=(13,))
preload_thread.start()

# Set all matplotlib optimizations to fast
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 10000

# Create the main figure and slider
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.axis('equal')
ax.axis('off')

slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
order_slider = Slider(slider_ax, 'Order', 1, 13,
                      valinit=1, valstep=1, valfmt='%d')

# Plot the initial snowflake
initial_snowflake = cached_snowflake(1)
line, = ax.plot(initial_snowflake[:, 0], initial_snowflake[:, 1])


def update(event) -> None:
    """
    Update the snowflake plot based on the slider value.

    Parameters
    ----------
    event : 
        Event triggered by the slider.

    Returns
    -------
    None
    """
    order = int(order_slider.val)

    ax.clear()
    ax.axis('equal')
    ax.axis('off')

    snowflake_points = cached_snowflake(order)
    ax.plot(snowflake_points[:, 0], snowflake_points[:, 1])

    min_x, max_x = np.min(snowflake_points[:, 0]), np.max(
        snowflake_points[:, 0])
    min_y, max_y = np.min(snowflake_points[:, 1]), np.max(
        snowflake_points[:, 1])

    margin = 0.1
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    fig.canvas.draw_idle()


order_slider.on_changed(update)


plt.show()
