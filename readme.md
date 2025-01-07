# README for Stereo Vision Depth Estimation with Block Matching

This project implements a stereo vision depth estimation pipeline using the **Block Matching** algorithm. It processes stereo images from the Middlebury dataset to generate disparity maps and depth maps, apply filtering techniques, and evaluate the performance of the generated depth maps using error metrics.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Directory Structure](#directory-structure)
5. [Usage](#usage)
6. [Outputs](#outputs)
7. [Error Metrics](#error-metrics)
8. [References](#references)

---

## **Overview**

Stereo vision allows for depth perception by analyzing the disparity between two images captured from slightly different viewpoints. This code:

- Computes disparity maps using OpenCV's `StereoBM` algorithm.
- Converts disparity maps to depth maps using calibrated stereo camera parameters.
- Applies multiple filtering techniques to improve disparity map quality.
- Compares the generated depth maps against ground truth data and evaluates performance using metrics like **MAE**, **RMSE**, **MAPE**, and **RMSPE**.

The Middlebury dataset is used as a benchmark for evaluation.

---

## **Features**

- **Disparity Map Computation:** Utilizes OpenCV's block matching algorithm.
- **Depth Map Generation:** Calculates depth maps from disparity maps using stereo calibration parameters.
- **Filtering Techniques:**
  - Median Filtering
  - Bilateral Filtering
  - Weighted Least Squares (WLS) Filtering
- **Error Metrics:** Compares predicted depth maps with ground truth depth maps.
- **Visualization:** Displays and saves disparity and depth maps with a colormap for better interpretation.

---

## **Requirements**

### Dependencies

- Python 3.7 or higher
- OpenCV (4.5.5 or later, including `cv2.ximgproc`)
- NumPy
- Middlebury Stereo Vision Dataset (images provided by the user)

### Installation

Install required Python libraries using `pip`:

```bash
pip install opencv-python opencv-contrib-python numpy
```

### Usage

Prepare Input Images: Place the left and right stereo images (bag1.png, bag2.png) and the ground truth depth map (bag_image.png) in the images folder.

Run the Script: Execute the script using:

bash

        Copy code
      python (name of the file).py

View Outputs: Processed disparity maps, depth maps, and error metrics will be saved in the image_depth/ folder.

Key outputs are also displayed during execution.

### Outputs

Disparity Maps:

Before Filtering (\_disparity_map_before_filter.png)
After Median Filtering (\_disparity_map_median_filter.png)
After Bilateral Filtering (\_disparity_map_bilateral_filter.png)
After WLS Filtering (\_disparity_map_wls_filter.png)
Depth Maps:

Depth maps corresponding to each filtering technique (_depth_map_{filter_name}.png)
Error Metrics:

Saved in error_metrics.txt:
MAE (Mean Absolute Error): Measures absolute differences.
RMSE (Root Mean Squared Error): Measures variance.
MAPE (Mean Absolute Percentage Error): Measures percentage differences.
RMSPE (Root Mean Squared Percentage Error): Squared percentage variance.

### Error Metrics

Calculated Metrics:
MAE: Mean Absolute Error (lower is better)
RMSE: Root Mean Squared Error (lower is better)
MAPE: Mean Absolute Percentage Error (lower is better)
RMSPE: Root Mean Squared Percentage Error (lower is better)
