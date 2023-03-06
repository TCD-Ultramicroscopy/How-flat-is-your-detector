# todo: make scipy import explicit, check units
import os
import numpy as np
from scipy.ndimage import rotate, shift

# local imports
import utils
from read_dm3 import DM3

#
# User inputs
#

# Path to a DigitalMicrograph file containing a detector map (stacks will be averaged)
detector_image_path = r""

# name of detector, purely for plotting and output
detector_name = "How flat is " + os.path.basename(detector_image_path)

# shift (x, y) of image to get center of detector in middle (in integer pixels)
center_shift = (0, 0)

# Rotation of detector map if desired (in degrees)
rotation = 0

# Threshold sets fraction of dynamic range above which is considered active region
threshold = 0.3

#
# Processing starts here
#

# Load the detector image
det_image = DM3(detector_image_path).image.astype(np.float64)

# If a stack, take the average through the stack
if det_image.ndim > 2:
    det_image = np.sum(det_image, axis=2)

# get the number of pixels, just for convenience
n = det_image.size

# get a number to fill the background with for shifting/rotating (takes an average of the lowest 4% of pixels)
n_low = int(0.04 * n)
fill = np.mean(np.sort(det_image.ravel())[:n_low])

# deciding contrast limits for determining active area (and for plotting)
pcnt_min = 0.5
pcnt_max = 0.1
n_min = int(n * pcnt_min/100)
n_max = int(n * pcnt_max/100)

if n_max == 0:
    n_max = 1

vmn = np.sort(det_image.ravel())[n_min]
vmx = np.sort(det_image.ravel())[-n_max]

# calculate the image shift to get on center
_xs = -int(center_shift[0])
_ys = -int(center_shift[1])

# shift to center
det_new = shift(det_image, shift=(_ys, _xs), cval=fill)

# rotate
det_new = rotate(det_new, -rotation, cval=fill)

# get mask for active area as 50% of dynamic range
mask_new = det_new > vmn + (vmx - vmn) * threshold

# calculate "how flat?" parameters
# This will plot data, save plot and print out how flat parameters
# Will aldo return parameters if further processing is wanted
utils.calculate_how_flat(det_new, mask_new, f'{detector_name}.pdf')