# P2020 Flicker metrics calculator

This is a reference implementation for IEEE P2020 flicker IQF metrics. The code is currently in beta testing - feel free
to download, use and provide feedback.

A full description of the P2020 flicker IQFs is available in the IEEE P2020 pre-release (https://www.techstreet.com/ieee/standards/ieee-p2020?gateway_code=ieee&vendor_id=6765&product_id=2505612)

The IQFs calculated by this code are:

- Flicker modulation index (FMI)
- Flicker detection index (FDI)
- Modulation mitigation probability (MMP)

![image](https://github.com/CARpublic/P2020-flicker/assets/141751829/557b8503-e904-41f5-9f38-c83143c3ab02)

![image](https://github.com/CARpublic/P2020-flicker/assets/141751829/e37be00e-b94a-40b9-9764-a9fa2330874f)

![image](https://github.com/CARpublic/P2020-flicker/assets/141751829/d052e87b-e2f1-4675-a147-e72459ba5a1e)


## Getting up and running
Use pip install -r requirements.txt to install the necessary dependencies. 

The code has been tested on Python 3.8.

Once the dependencies are installed, flicker IQFs can be calculated as follows:

### Part 1
Define ROIs. For the sample videos, an Image Engineering DTS target is used, but the code can easily be adapted to work 
with any target. 
Use the sample script "def_rois_demo.py" to manually define measurement ROIs. The selected ROI locations are stored to a
.pkl file

### Part 2
Calculate P2020 metrics
- FMI, FDI, MMP all calculated
- Saturation warning is included
- Save to csv file
- Save output figures

To Do:
- Read in target luminance file and plot results vs luminance
- Code reviews, etc
