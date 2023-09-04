import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import cv2
from flicker.metrics import calc_FMI, calc_FDI, calc_MMP, saturation_warning

def calc_roi_mean(roi_file, input_video):
    """
    This function takes in an ROI file definition, input video, and returns a matrix of mean values for each roi
    :param roi_file:
    :param input_video:
    :return roi_means:
    """

    # load pickle file containing ROIs
    df = pd.read_pickle(roi_file).astype(np.uint16)

    # get the number of ROIs
    num_rois = df.shape[0]

    col_headings = []
    for i in range(0, num_rois):
        col_headings.append('ROI_' + str(i))

    roi_means = np.empty([num_rois, 1])

    # load video file
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    # display ROIs on first frame
    ret, img = cap.read()
    plt.figure()
    plt.imshow(img)
    plt.imshow(img)
    ax = plt.gca()

    for i in range(num_rois):
        rect = patches.Rectangle((df['x'][i], df['y'][i]), df['roi_width'][i], df['roi_height'][i], linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title('ROIs defined by user')
    plt.show()

    count = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            temp = np.zeros((num_rois, 1))
            for i in range(num_rois):
                temp[i] = np.mean(img[df['y'][i].astype(np.uint16):(df['y'][i] + df['roi_height'][i]).astype(np.uint16),
                                  df['x'][i].astype(np.uint16):(df['x'][i] + df['roi_width'][i]).astype(np.uint16)])

            roi_means = np.concatenate((roi_means, temp), axis=1)
            count = count + 1
            print("Frame number: ", count)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(roi_means.T, columns=col_headings)

    return df

def save_to_file(filename, FMI, FDI, MMP, sat_flag):
    """
    Function to save results to csv file
    :param filename:
    :param FMI:
    :param FDI:
    :param MMP:
    :param sat_flag:
    :return:
    """

    output = pd.DataFrame()

    output['FMI'] = FMI
    output['FDI'] = FDI
    output['MMP'] = MMP
    output['sat_flag'] = sat_flag
    output.to_csv(filename)


def main():
    print('Calculate Flicker metrics')

    # read in input file
    input_video = "sample videos//test_vid1.mp4"

    # read in pre-defined ROI locations
    roi_means = calc_roi_mean(roi_file='rois_flicker_measurement.pkl', input_video=input_video)

    # start index of measurement (before this point is the "baseline", no flicker data
    start_idx = 100

    # calculate flicker metrics
    FMI = calc_FMI(roi_means, start_index=start_idx)

    FDI = calc_FDI(roi_means, start_index=start_idx, contrast_threshold=0.1)

    MMP = calc_MMP(roi_means, start_index=start_idx, delta=0.1)

    sat_flag, sat_idx = saturation_warning(roi_means, max_val=240)

    # save flicker metrics to file
    save_to_file('figures/output.csv', FMI, FDI, MMP, sat_flag)

    # plot results, save figures to file
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot()
    ax1.plot(roi_means)
    ax1.set_title('Time series of all ROIs')
    ax1.set_xlabel('frame number')
    ax1.set_ylabel('DN')
    if sat_flag.any():
        ax1.text(0, 240, '*Saturation warning', fontsize=12, color='red')

    plt.savefig('figures//time_series.png')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    plt.plot(FMI)
    ax2.set_title('Flicker Modulation Index (FMI)')
    ax2.set_xlabel('ROI')
    ax2.set_ylabel('FMI')
    ax2.set_ylim([0, 1.2])
    if sat_flag.any():
        ax2.text(0, 1, '*Saturation warning', fontsize=12, color='red')
    plt.savefig('figures//FMI.png')

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot()
    plt.plot(FDI)
    ax3.set_title('Flicker Detection Index (FDI)')
    ax3.set_xlabel('ROI')
    ax3.set_ylabel('FDI')
    ax3.set_ylim([0, 1.2])
    if sat_flag.any():
        ax3.text(0, 1, '*Saturation warning', fontsize=12, color='red')
    plt.savefig('figures//FDI.png')

    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot()
    plt.plot(MMP)
    ax4.set_title('Modulation Mitigation Probability (MMP)')
    ax4.set_xlabel('ROI')
    ax4.set_ylabel('MMP')
    plt.ylim([0, 1.2])
    if sat_flag.any():
        ax4.text(0,1,'*Saturation warning', fontsize=12, color='red')
    plt.savefig('figures//MMP.png')

    plt.show()

if __name__ == "__main__":
    main()
