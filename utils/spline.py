import numpy as np

def interpolate_nc_spline(control_points, num_steps=10):
    """
    Calculates points along a natural cubic spline from control points.

    This function takes 0-1 normalized control points (where the origin is at
    the bottom-left) and interpolates them to generate a smooth curve.

    Args:
        control_points (list[dict]): [{'x': 0, 'y': 0}, ..., {'x': 1, 'y': 1}]
        num_steps (int): The total number of points to generate for the spline.

    Returns:
        list[tuple]: A list of (x, y) tuples representing the interpolated
                     points on the spline.
    """
    if not control_points or len(control_points) < 2:
        return []

    # Sort points by x-coordinate
    points = sorted(control_points, key=lambda p: p['x'])

    x = np.array([p['x'] for p in points])
    y = np.array([p['y'] for p in points])
    n = len(points) - 1

    if np.any(np.diff(x) <= 0):
        return []

    # Direct translation of the JavaScript `calculateNaturalCubicSpline`
    # function used on the client spline component.

    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    c = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c[n] = 0
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]

    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - h * (c[1:] + 2 * c[:-1]) / 3

    # Evaluate the spline at the desired steps
    spline_points = []
    x_out = np.linspace(x[0], x[-1], num_steps)

    for x_val in x_out:
        # Find the correct spline segment for the given x_val
        segment_index = np.searchsorted(x, x_val, side='right') - 1
        segment_index = max(0, min(segment_index, n - 1))

        dx = x_val - x[segment_index]
        y_val = (
            y[segment_index] +
            b[segment_index] * dx +
            c[segment_index] * dx**2 +
            d[segment_index] * dx**3
        )
        spline_points.append((x_val, y_val))

    return spline_points