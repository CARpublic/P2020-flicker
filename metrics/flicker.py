import numpy as np


def calc_FMI(roi_means, start_index):
    """
    Pass in roi_means dataframe, calculate FMI for each ROI
    :param roi_means: time series of ROI mean values
    :param start_index: frame index for start of calculation
    :return FMI:
    """

    # get the number of ROIs
    num_rois = roi_means.shape[0]

    fmi_max = np.max(roi_means[start_index:], axis=0)
    fmi_min = np.min(roi_means[start_index:], axis=0)

    FMI = (fmi_max - fmi_min) / (fmi_max + fmi_min)

    return FMI

def calc_FDI(roi_means, start_index, contrast_threshold=0.1):
    """

    :param roi_means: time series of ROI mean values
    :param start_index: frame index for start of calculation
    :param contrast_threshold: threshold for FDI calculation.
    :return FDI:
    """

    x_ref_off = np.mean(roi_means[:np.divide(start_index,4).astype(np.uint16)], axis=0)

    roi_means = roi_means[start_index:]

    num_samples = roi_means.shape[0]

    flicker_signal = (roi_means - x_ref_off) / x_ref_off

    FDI = (flicker_signal > contrast_threshold).sum() / num_samples

    return FDI

def calc_MMP(roi_means, start_index, delta=0.1):
    """
    calculates MMP
    :param roi_means: time series of ROI mean values
    :param start_index: frame index for start of calculation
    :param contrast_threshold: user defined contrast threshold for MMP calculation
    :return MMP:
    """

    roi_means = roi_means[start_index:]

    num_samples = roi_means.shape[0]

    x_ref_mmp = np.mean(roi_means, axis=0)

    flicker_signal = (roi_means > (x_ref_mmp - delta*x_ref_mmp)) & (roi_means < (x_ref_mmp + delta*x_ref_mmp))

    MMP = flicker_signal.sum() / num_samples

    return MMP

def saturation_warning(roi_means, max_val):
    """
    If pixel values saturated (i.e. clip), flicker measurements are affected. This function detects if clipping happens,
    and returns a flag indicating saturation has happened, along with the index of saturated frames
    :param roi_means: time series calculated from mean of ROIs
    :param max_val: threshold for saturation
    :return sat_flag, sat_idx:
    """
    sat_flag = False
    sat_idx = np.zeros(np.shape(roi_means), dtype=bool)

    sat_idx = np.greater(roi_means, max_val)

    sat_flag = sat_idx.any()

    return sat_flag, sat_idx
