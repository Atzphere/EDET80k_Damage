'''
This module contains code to account for the warped perspective
caused by aiming the camera off-axis to avoid measuring its own reflection.
It uses a set of manually determined calibration points stored in REFERENCE_PATH
to generate an image/coordinate transformation which maps points on the camera image
to points within the chip coordinate system.


'''
import cv2
import numpy as np
from collections.abc import Iterable
import os

os.chdir(os.path.dirname(__file__))
REFERENCE_PATH = "./CameraPerspectivePoints.txt"

# EDIT THESE FOR RECALIBRATION

# approximate WIDTH and HEIGHT (pixels) of the rectangular feature used for calibration.
# as of september 2024 we are using the outline of the aluminum chip-holding brick.

WIDTH = 240
HEIGHT = 188

CHIP_DIM = (32, 32)  # physical dimensions of the chip
CHIP_ORIGIN = (154, 260)  # sensor bottom-left corner in image
CHIP_EXTREMUM = (307, 110)  # sensor top-right corner

# used to inverting axes
OPTRIS_IMGHEIGHT = 288


# "offset"; the position of a reference in image coordinates
# This is currently the top-left corner of the rectangle.
# this ensures the transformed image is aligned; this point will remain unmoved
ox = 73
oy = 83

# END CONSTANTS

# in-image positions of the rectangle's corners
actual = np.loadtxt(REFERENCE_PATH, delimiter=",", dtype="float32")

# the new, "squared" positions that you want your corners to be at
target = np.float32([[ox, oy], [ox + WIDTH, oy],
                     [ox, oy + HEIGHT], [ox + WIDTH, oy + HEIGHT]])

# build transformation matrix
image_transform = cv2.getPerspectiveTransform(actual, target)


def cv_xy_package(x, y):
    '''
    package a point and/or sequences of coordinates corresponding to points
    in an opencv-digestible manner
    '''

    if isinstance(x, Iterable):
        pts = np.array([np.zeros((len(x), 2))], dtype="float32")
        pts[:, :, 0] = x
        pts[:, :, 1] = y

    else:  # single value case
        pts = np.array([[[x, y]]], dtype="float32")

    return pts


def perspective_map_points(x, y, transform):
    '''
    Wrapper function for PerspectiveTransform to be more flexible with inputs
    Returns an x, y of newly transformed points

    Params:
    x, y: float or Iterable[float]: Coordinate or iterable of coordinates along
    either the x or y axis. x and y should have the same length.

    transform: the transformation matrix to be applied to x and y.
    '''
    # format points for cv2

    pts = cv_xy_package(x, y)

    # apply the perspective transform to the points
    transformed = cv2.perspectiveTransform(pts, m=transform)
    xnew, ynew = transformed[:, :, 0], transformed[:, :, 1]
    return xnew, ynew


def image_coords_to_cartesian(x, y, imgheight=OPTRIS_IMGHEIGHT):
    '''
    Converts image pixel coordinates (y=0 at top) to cartesian (y=0 at bottom).
    '''

    return x, -y + imgheight


def to_arbitrary_coords(x, y, dimensions, origin, extremum):
    '''

    Using a reference origin and extremum point defining a rectangular ROI
    as well as its physical dimensions, take arbitrary points and map them
    to physical space in a cartesian coordinate system centered on the origin.

    params:
    x, y float or Array[float]: x and y coordinates (should be paired) to be converted

    dimensions Tuple(float, float): reference dimensions describing the width and height of
                                    a rectangular ROI.

    origin, extremum Tuple(float, float): reference points to define the new coordinate system with.
                                          the origin will be (0, 0), and the extremum will be at the max dimensions
                                          of the rectangle.
    '''
    # shift everything so that (0, 0) is the origin

    ox, oy = origin
    ex, ey = extremum

    x, y = x - ox, y - oy
    ex, ey = ex - ox, ey - oy

    # normalize everything out of the shifted extremum and then scale by desired dimensions

    dim_x, dim_y = dimensions

    x = (x / ex) * dim_x
    y = (y / ey) * dim_y

    return x[0], y[0]


def camera_to_roi(x, y, transform=image_transform, roi_dim=CHIP_DIM, origin=CHIP_ORIGIN, extremum=CHIP_EXTREMUM, imgheight=OPTRIS_IMGHEIGHT):
    '''
    Wraps the entire mapping process into a convenient function.
    Default values can be configured at the top of the file.
    '''
    # pre-calibrate by correcting for barrel distortion
    points = cv_xy_package(x, y)

    # undistorted = cv2.undistortPoints(points, cam, distCoeff)
    # x_prec, y_prec = undistorted[:, :, 0], undistorted[:, :, 1]

    # perspective map
    x_trans, y_trans = perspective_map_points(x, y, transform)

    # change y axis direction
    x_flip, y_flip = image_coords_to_cartesian(x_trans, y_trans, imgheight)

    # do the same thing with the reference points
    ox, oy = origin
    ex, ey = extremum

    # for now, distortion correction is disabled - does weird things to coordinates
    # undistorted = cv2.undistortPoints(points, cam, distCoeff)
    # x_prec, y_prec = undistorted[:, :, 0].flatten(), undistorted[:, :, 1].flatten()

    xref_trans, yref_trans = perspective_map_points(
        [ox, ex], [oy, ey], transform)
    refx_flip, refy_flip = image_coords_to_cartesian(
        xref_trans, yref_trans, imgheight)

    refx_flip = refx_flip[0]
    refy_flip = refy_flip[0]
    origin_converted = (refx_flip[0], refy_flip[0])
    extremum_converted = (refx_flip[1], refy_flip[1])

    # convert everything to chip coordinates
    cx, cy = to_arbitrary_coords(
        x_flip, y_flip, roi_dim, origin_converted, extremum_converted)

    # return cx, cy
    return x_trans[0][0], y_trans[0][0]
