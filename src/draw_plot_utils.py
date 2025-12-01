import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def draw_pose(img, pose, color):
    x, y, theta = pose
    x = int(x)
    y = int(y)

    # robot center
    cv2.circle(img, (x, y), 5, color, -1)

    # heading line
    x2 = int(x + 20 * np.cos(theta))
    y2 = int(y + 20 * np.sin(theta))
    #cv2.line(img, (x, y), (x2, y2), color, 2)
    cv2.arrowedLine(img, (x, y), (x2, y2), color, thickness=1, tipLength=0.3)

def draw_path(img, path, color):
    for idx in range((len(path))-1):
        # put in correct frame
        y, x = path[idx]
        y_after, x_after = path[idx + 1]
        x = int(x)
        y = int(y)
        x_after = int(x_after)
        y_after = int(y_after)
        cv2.circle(img, (x, y),radius=1, color=0)
        cv2.line(img, (x, y), (x_after, y_after), color, thickness=1)

def draw_point(img, point, color):
    x, y = point
    x = int(x)
    y = int(y)
    cv2.circle(img, (x, y), 4, color, -1)


def plot_covariance_ellipse(mean, cov, ax=None, n_std=2.0, **kwargs):
    """
    Plot an ellipse representing a 2D covariance matrix.

    mean: array-like, shape (2,) — the center of the ellipse  
    cov: 2×2 covariance matrix  
    ax: matplotlib Axes (optional)  
    n_std: number of standard deviations (radius)  
    kwargs: passed to Ellipse patch
    """
    if ax is None:
        ax = plt.gca()

    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Width and height (2 * sqrt(lambda) * n_std)
    width, height = 2 * n_std * np.sqrt(vals)
    # Angle of rotation (in degrees) of the largest eigenvector
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    return ellipse