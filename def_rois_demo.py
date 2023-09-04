import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import numpy as np
import tifffile
from matplotlib.backend_bases import MouseButton

import matplotlib.patches as patches

# read in input file. At the moment, this is just a tiff file. Long term, will add video frame input
img = cv2.imread("figures//simulated_image_0.png")

# define the number of ROIs
num_rois = 36
# +/- size of ROI
roi_width = 20
roi_height = 20

# use the ginput function to get the user to define the centre point of each ROI
plt.figure()
plt.title('Use ginput function to define the centre of each ROI')
plt.imshow(img)

pts = []

# ginput function to manually define a single point to be used to define ROI location
pts = np.asarray(plt.ginput(num_rois, timeout=-1, show_clicks=True))

# define array to hold the ROI definition [x, y, roi_width, roi_height]
box_coords = np.empty((num_rois, 4))

for i in range(num_rois):
    box_coords[i:] = np.asarray([pts[i,0]-roi_width/2,pts[i,1]-roi_height/2, roi_width, roi_height])

# add to a dataframe
df = pd.DataFrame(box_coords,columns=['x', 'y', 'roi_width','roi_height'])

# Plot ROIs on image
plt.figure()
plt.imshow(img)
ax = plt.gca()

for i in range(num_rois):
    rect = patches.Rectangle((df['x'][i],df['y'][i]),df['roi_width'][i],df['roi_height'][i],linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


plt.title('ROIs defined by user')
plt.show()

# save dataframe to pickle file
pd.to_pickle(df,"rois.pkl")





