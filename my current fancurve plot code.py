import numpy as np

from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class FanCurve:
    """
    Production-quality fan curve representation.

    Stores raw points and produces a smooth,
    physically realistic curve suitable for:

    - Fan maps
    - System curve intersections
    - Duty point calculation
    - Export to plotting libraries
    """

    def __init__(
        self,
        points,
        name="",
        smooth_factor=0.003,
        resolution=2000,
    ):
        self.name = name

        self.raw_points = np.asarray(points, dtype=float)

        self.smooth_factor = smooth_factor
        self.resolution = resolution

        self._clean_points()

        self._fit()

    # ---------------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------------
    def _clean_points(self):

        pts = self.raw_points

        keep = [True]

        for i in range(1, len(pts)):
            dx = pts[i, 0] - pts[i - 1, 0]
            dy = pts[i, 1] - pts[i - 1, 1]

            keep.append(np.hypot(dx, dy) > 1e-6)

        pts = pts[np.array(keep)]

        # Normalize direction
        if pts[0, 1] > pts[-1, 1]:
            pts = pts[::-1]

        self.points = pts


    def plot(self, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        x, y = self.get_curve()

        ax.plot(x, y, **kwargs)

    @property
    def length(self):

        p = self.points

        dx = np.diff(p[:, 0])
        dy = np.diff(p[:, 1])

        return np.sum(np.hypot(dx, dy))

    # ---------------------------------------------------------
    # SPLINE FIT
    # ---------------------------------------------------------

    def _fit(self):

        pts = self.points

        x = pts[:, 0]
        y = pts[:, 1]

        if len(x) < 4:

            self.x_smooth = x
            self.y_smooth = y

            return

        s = self.smooth_factor * self.length * len(x)

        tck, u = splprep(
            [x, y],
            s=s,
            k=min(3, len(x) - 1),
        )

        u_dense = np.linspace(
            0,
            1,
            self.resolution,
        )

        xs, ys = splev(u_dense, tck)

        self.tck = tck
        self.u = u_dense

        self.x_smooth = np.asarray(xs)
        self.y_smooth = np.asarray(ys)

    # ---------------------------------------------------------
    # CURVATURE
    # ---------------------------------------------------------

    def curvature(self):

        x = self.x_smooth
        y = self.y_smooth

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        k = np.abs(
            dx * ddy - dy * ddx
        ) / np.maximum(
            (dx * dx + dy * dy) ** 1.5,
            1e-12,
        )

        return k

    def mean_curvature(self):

        return float(np.mean(self.curvature()))

    # ---------------------------------------------------------
    # ERROR ANALYSIS
    # ---------------------------------------------------------

    def fitting_error(self):

        raw = self.points

        smooth = np.column_stack(
            [self.x_smooth, self.y_smooth]
        )

        distances = cdist(raw, smooth)

        return np.min(
            distances,
            axis=1,
        )

    def rms_error(self):

        err = self.fitting_error()

        return float(
            np.sqrt(
                np.mean(err ** 2)
            )
        )

    # ---------------------------------------------------------
    # INTERSECTION
    # ---------------------------------------------------------

    def intersection(self, other_curve):

        a = np.column_stack(
            [self.x_smooth, self.y_smooth]
        )

        b = np.column_stack(
            [other_curve.x_smooth,
             other_curve.y_smooth]
        )

        d = cdist(a, b)

        idx = np.unravel_index(
            np.argmin(d),
            d.shape,
        )

        p1 = a[idx[0]]
        p2 = b[idx[1]]

        return (p1 + p2) / 2

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------

    def get_curve(self):

        return (
            self.x_smooth.copy(),
            self.y_smooth.copy(),
        )

    def get_raw(self):

        return self.points.copy()

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------

    def summary(self):

        return {
            "name": self.name,
            "points": len(self.points),
            "length": self.length,
            "rms_error": self.rms_error(),
            "mean_curvature": self.mean_curvature(),
        }