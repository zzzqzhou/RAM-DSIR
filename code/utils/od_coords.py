
""" Functions to deal with OD and Fovea localization. """

import numpy as np

import matplotlib.pyplot as plt

from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.feature import peak_local_max

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_fill_holes

def find_od_f(pred):

    local_max = maximum_filter(pred, size=5, mode='constant')
    coordinates = peak_local_max(pred, min_distance=50, num_peaks=2)

    return coordinates

def plot_coords(img, coords):

    plt.imshow(img)
    plt.plot(coords[:, 1], coords[:, 0], 'r.')


def get_new_peaks(coords, shp):

    xo, yo = shp
    xp, yp = coords

    xn = (xp * xo) / 512
    yn = (yp * yo) / 512
    coord_new = (xn, yn)

    return coord_new


def distance_metric(pred_coords, orig_coords):

    xp, yp = pred_coords
    xo, yo = orig_coords

    dist = np.sqrt((xo - xp) ** 2 + (yo - yp) ** 2)

    return dist


def distance_error(pred_coords, orig_coords, od_radius=88., r=1):
    xp, yp = pred_coords
    xo, yo = orig_coords

    dist = np.sqrt((xo - xp) ** 2 + (yo - yp) ** 2)

    error_od_radius = dist / (od_radius * r)

    return dist, error_od_radius


def determine_od(image, coords, neigh=3):
    """ Determines which peak corresponds to the OD and to the Fovea.
    input params:
        image: the RGB image
        coords: the coordinates of the two selected peak_coords
        neigh: the neighbourhood to consider for evaluation
    returns:
        od_coords: the coordinates of the peak selected as OD
        fov_coords: the coordinates of the peak selected as Fovea
    """
    # create a special case for the border, in case the peak is located close
    # to it, it must always have neighbours
    coords[np.where(coords < neigh)] = neigh
    coords[np.where(coords > (511-neigh))] = (511-neigh)

    coord_new1, coord_new2 = coords[0], coords[1]

    # Calculate the mean intensity of each peak and its neighbohood
    i1 = np.mean(image[:,:,1][coord_new1[0]-neigh:coord_new1[0]+neigh,
                                coord_new1[1]-neigh:coord_new1[1]+neigh])
    i2 = np.mean(image[:,:,1][coord_new2[0]-neigh:coord_new2[0]+neigh,
                                coord_new2[1]-neigh:coord_new2[1]+neigh])

    # The OD is expected to have higher intensity
    if i1 >= i2:
        od_coords = coord_new1
        fov_coords = coord_new2

    elif i1<i2:
        od_coords = coord_new2
        fov_coords = coord_new1
    else:
        od_coords = (256,256)
        fov_coords = (256,256)

    return od_coords, fov_coords

def get_diameters(od_mask):
    """ Function that returns the internal diameters of the optic disk segmentations"""
    collapsed = np.sum(od_mask, axis=0)
    # These indices will be already sorted
    indices = np.where(collapsed == collapsed.max())[0]
    c = indices[int(round((len(indices) - 1) / 2))]
    indices = np.where(collapsed >0)[0]
    cmin = indices[0]
    cmax = indices[-1]

    # Same for rows
    collapsedr = np.sum(od_mask, axis=1)
    # These indices will be already sorted
    indices = np.where(collapsedr == collapsedr.max())[0]
    r = indices[int(round((len(indices) - 1) / 2))]
    indices = np.where(collapsedr>0)[0]
    rmin = indices[0]
    rmax = indices[-1]

    dc = cmax - cmin
    dr = rmax - rmin
    return dc, dr

def get_centroid(mask, fill=True):
    """
    Function that retuns the coordinates of the centroid of the OD or fovea
    """
    if fill is True:
        mask = binary_fill_holes(mask)

    collapsedc = np.sum(mask, axis=0)
    indices = np.where(collapsedc == collapsedc.max())[0]
    c = indices[int(round((len(indices) - 1) / 2))]

    collapsedr = np.sum(mask, axis=1)
    indices = np.where(collapsedr == collapsedr.max())[0]
    r = indices[int(round((len(indices) - 1) / 2))]

    return c, r


def get_peak_coordinates(image, threshold=0.2):
    image_gray = rgb2gray(image)
    image_gray = np.pad(image_gray, (15, 15), 'constant')

    blobs = blob_log(image_gray, min_sigma=10, max_sigma=50, threshold=threshold)
    bb = blobs[:, :2].astype('int')

    if blobs.shape[0] < 2:
        new_blobs = np.copy(blobs)

        while new_blobs.shape[0] < 2:

            threshold = 0.8 * threshold
            print(threshold)
            if threshold < 0.001:
                print('Threshold too low! Passing...')
                break
            else:
                new_blobs = blob_log(image, min_sigma=10, max_sigma=50,
                                     threshold=threshold)

        blobs = new_blobs
        print(blobs.shape)
        if blobs.shape[0] < 2:
            np.concatenate((blobs, [[256, 256, 0]]), axis=0)

    blobs = blobs - 15 # to account for to the initial padding
    blobs[np.where(blobs > 512)] = 0
    blobs[np.where(blobs < 0)] = 0


    blobs = blobs[:, :2].astype('int')

    bb2 = blobs[:, :2].astype('int')

    #if blobs.shape[0] > 2:
    #    sorted_indx = np.argsort(image[bb2[:, 0], bb2[:, 1]], axis=None)[::-1]
    #    print sorted_indx
    #    blobs = bb2[sorted_indx[:2]]
    return blobs