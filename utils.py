from skimage.transform import warp_polar
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, medfilt

def to_polar(image):
    # center is assumed to be center of image, we have already corrected this
    polar = warp_polar(image)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(image, cmap='magma')
    # axs[1].imshow(polar, cmap='magma')
    # plt.show()

    return polar

def calculate_background(image, mask):
    # get fraction of background to account for detectoe edges etc
    mn = image[np.logical_not(mask)].min()
    mx = image[np.logical_not(mask)].max()
    lvl = mn + (mx - mn) / 4

    return np.mean(image, where=image < lvl)

def calculate_active_average(image, mask):
    return np.mean(image, where=mask)

def find_ellipticity(inner_edge):

    elp = 0

    for i in range(inner_edge.shape[0]):

        inds = [i, i + 90, i + 180, i + 270]
        for ii in range(len(inds)):
            if inds[ii] >= 360:
                inds[ii] -= 360

        axis_1 = inner_edge[inds[0]] + inner_edge[inds[2]]
        axis_2 = inner_edge[inds[1]] + inner_edge[inds[3]]

        _elp = axis_1 / axis_2
        if _elp > elp:
            elp = _elp

    return elp

def calculate_smoothness(image, mask, hist):
    # # method 1, find max then fwhm
    # hist_f = hist[0]
    # bin_width = hist[1][1] - hist[1][0]
    # max_i = np.argmax(hist_f)
    # above_hm = hist_f > hist_f[max_i] / 2

    # for i in range(max_i):
    #     if not above_hm[max_i - i]:
    #         break
    #     low = max_i - i

    # for i in range(hist_f.size - max_i):
    #     if not above_hm[max_i + i]:
    #         break
    #     high = max_i + i

    # fwhm_1 = (high - low) * bin_width

    # method 2, just stats
    std = np.std(image, where=mask)
    fwhm_2 = std * 2.355

    return fwhm_2


def calculate_how_flat(_det_image, _mask_image, fname=None):

    det_image = _det_image.copy()
    mask_image = _mask_image.copy()

    det_background = calculate_background(det_image, mask_image)

    det_image -= det_background

    det_image /= calculate_active_average(det_image, mask_image)

    det_polar = to_polar(det_image)
    mask_polar = to_polar(mask_image).astype(bool) # this is where there is active detector
    rad_mask = to_polar(np.ones_like(det_image)).astype(bool) # this is just where there is data

    det_az = np.mean(det_polar, axis=1, where=mask_polar)

    det_rad = np.mean(det_polar, axis=0, where=rad_mask)
    mask_rad = np.mean(mask_polar, axis=0, where=rad_mask) > 0.9

    det_hist_raw = np.histogram(det_image[mask_image], bins=50)

    det_hist = medfilt(det_hist_raw[0]), det_hist_raw[1]

    # calculate radius of inner
    inner_r = np.argmax(mask_polar, axis=1)
    inner_r_filt = savgol_filter(inner_r, 51, 3)

    ellipticity = find_ellipticity(inner_r_filt)

    if ellipticity == 0:

        inner_r = np.argmax(np.logical_not(mask_polar), axis=1)
        inner_r_filt = savgol_filter(inner_r, 51, 3)

        ellipticity = find_ellipticity(inner_r_filt)


    # calculate flatness
    det_flatness = np.std(det_rad, where=mask_rad)

    # calculate roundness
    det_roundness = np.std(det_az)

    # calculate smoothness
    det_smoothness = calculate_smoothness(det_image, mask_image, det_hist)

    print("Flatness")
    print(f"Analog: {det_flatness}")
    print("--------------------")

    print("Roundness")
    print(f"Analog: {det_roundness}")
    print("--------------------")

    print("Smoothness")
    print(f"Analog: {det_smoothness}")
    print("--------------------")

    print("Ellipticity")
    print(f"{(ellipticity - 1)}")
    print("--------------------")


    fig, axs = plt.subplots(2, 4, figsize=(11,5))

    axs[0, 0].imshow(mask_image * _det_image)
    axs[0, 1].imshow(mask_image)

    axs[0, 2].imshow(det_polar)
    axs[0, 3].imshow(rad_mask + 2 * mask_polar)
    axs[0, 3].plot(inner_r_filt, np.arange(det_polar.shape[0]), color='r', label="Inner")

    # ellipticity
    axs[1, 0].plot(inner_r, label="Raw")
    axs[1, 0].plot(inner_r_filt, label="Filtered")

    # flatness
    rad_start = np.argmax(mask_rad)
    rad_end = mask_rad.size - np.argmax(mask_rad[::-1]) - 1

    axs[1, 1].axvline(x=rad_start, color='#CCCCCC', label="Limits")
    axs[1, 1].axvline(x=rad_end, color='#CCCCCC')

    axs[1, 1].plot(det_rad)

    # roundness
    axs[1, 2].plot(det_az)

    # smoothness

    diffs = np.diff(det_hist_raw[1])

    x = [det_hist_raw[1][0]]
    y = [0]

    i = 0
    for c in range(diffs.size):
        x.append(x[i])
        y.append(det_hist_raw[0][c])

        i += 1

        x.append(x[i] + diffs[c])
        y.append(y[i])

        i += 1

    x.append(x[i])
    y.append(0)

    axs[1, 3].plot(x, y)

    #
    # make things look nice
    #

    axs[0, 3].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')

    axs[0, 0].set_title('Map')
    axs[0, 1].set_title('Mask')
    axs[0, 2].set_title('Map (polar)')
    axs[0, 3].set_title('Mask (polar)')

    axs[1, 0].set_xlabel('Azimuth (degrees)')
    axs[1, 1].set_xlabel('Radius (pixels)')
    axs[1, 2].set_xlabel('Azimuth (degrees)')
    axs[1, 3].set_xlabel('Normalised intensity')

    axs[1, 0].set_ylabel('Radius (pixels)')
    axs[1, 1].set_ylabel('Normalised Intensity')
    axs[1, 2].set_ylabel('Normalised Intensity')
    axs[1, 3].set_ylabel('Frequency')

    axs[1, 0].set_title('Inner radius')
    axs[1, 1].set_title('Azimuthal average')
    axs[1, 2].set_title('Radial average')
    axs[1, 3].set_title('Active area histogram')

    #
    # show plot
    #

    plt.tight_layout()

    if fname is not None:
        plt.savefig(f'{fname}')
    plt.show()
    plt.close()

    return det_flatness, det_roundness, det_smoothness, ellipticity

