import cv2
import numpy as np
import os

# Camera parameters from the new calibration data
cam0 = np.array([[7190.247, 0, 1938.523],
                 [0, 7190.247, 913.94],
                 [0, 0, 1]])
cam1 = np.array([[7190.247, 0, 2293.672],
                 [0, 7190.247, 913.94],
                 [0, 0, 1]])
doffs = 355.149  # Disparity offset in pixels
baseline = 174.945  # Baseline in millimeters
vmin = 32
vmax = 224

# Convert baseline from millimeters to meters for depth calculation
baseline_meters = baseline / 1000.0

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the images
left_image_filename = 'bag1.png'
right_image_filename = 'bag2.png'
ground_truth_filename = 'bag_image.png'  # Adjust this to your ground truth file name
left_image_path = os.path.join(script_dir, 'images', left_image_filename)
right_image_path = os.path.join(script_dir, 'images', right_image_filename)
ground_truth_path = os.path.join(script_dir, 'images', ground_truth_filename)

# Debugging: Print the paths
print(f"Loading left image from: {left_image_path}")
print(f"Loading right image from: {right_image_path}")
print(f"Loading ground truth depth map from: {ground_truth_path}")

# Check if files exist
if not os.path.exists(left_image_path):
    raise ValueError(f"Left image file not found: {left_image_path}")
if not os.path.exists(right_image_path):
    raise ValueError(f"Right image file not found: {right_image_path}")
if not os.path.exists(ground_truth_path):
    raise ValueError(f"Ground truth file not found: {ground_truth_path}")

# Load stereo images from files
left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)
ground_truth_depth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)  # Ground truth should be grayscale

# Ensure images were loaded successfully
if left_image is None:
    raise ValueError(f"Failed to load left image from {left_image_path}")
if right_image is None:
    raise ValueError(f"Failed to load right image from {right_image_path}")
if ground_truth_depth is None:
    raise ValueError(f"Failed to load ground truth depth map from {ground_truth_path}")

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Parameters for BM (Block Matching)
ndisp = 256  # Updated to match calibration data
block_size = 11  # This can be adjusted based on performance

# Create a stereo block matcher object using StereoBM with adjusted parameters
stereo = cv2.StereoBM_create(numDisparities=ndisp, blockSize=block_size)

# Compute the disparity map
disparity_left = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Convert the disparity map to an 8-bit image for visualization and filtering
disparity_left_8bit = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_left_8bit = np.uint8(disparity_left_8bit)

# Create the output directory if it doesn't exist
output_dir = os.path.join(script_dir, 'image_depth')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save and display the disparity map before filtering
output_path_disparity_before = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_before_filter.png")
cv2.imwrite(output_path_disparity_before, disparity_left_8bit)
cv2.imshow('Disparity Map (Before Filtering)', disparity_left_8bit)
cv2.waitKey(1)  # Wait briefly to ensure the image is displayed

# Apply median filtering to the 8-bit disparity map
disparity_left_median_filtered = cv2.medianBlur(disparity_left_8bit, 7)  # Ensure kernel size is odd

# Save and display the median filtered disparity map
output_path_disparity_median = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_median_filter.png")
cv2.imwrite(output_path_disparity_median, disparity_left_median_filtered)
cv2.imshow('Disparity Map (Median Filter)', disparity_left_median_filtered)
cv2.waitKey(1)

# Apply bilateral filtering to the 8-bit disparity map
disparity_left_bilateral_filtered = cv2.bilateralFilter(disparity_left_8bit, d=9, sigmaColor=75, sigmaSpace=75)

# Save and display the bilateral filtered disparity map
output_path_disparity_bilateral = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_bilateral_filter.png")
cv2.imwrite(output_path_disparity_bilateral, disparity_left_bilateral_filtered)
cv2.imshow('Disparity Map (Bilateral Filter)', disparity_left_bilateral_filtered)
cv2.waitKey(1)

# Apply WLS filtering to the disparity map
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
right_matcher = cv2.ximgproc.createRightMatcher(stereo)
disparity_right = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
disparity_wls_filtered = wls_filter.filter(disparity_left, left_gray, disparity_map_right=disparity_right)

# Normalize the WLS filtered disparity map for visualization
disparity_wls_filtered_8bit = cv2.normalize(disparity_wls_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_wls_filtered_8bit = np.uint8(disparity_wls_filtered_8bit)

# Save and display the WLS filtered disparity map
output_path_disparity_wls = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_wls_filter.png")
cv2.imwrite(output_path_disparity_wls, disparity_wls_filtered_8bit)
cv2.imshow('Disparity Map (WLS Filter)', disparity_wls_filtered_8bit)
cv2.waitKey(1)

# Function to calculate and save depth map
def calculate_and_save_depth_map(disparity_filtered, filter_name):
    disparity_filtered = disparity_filtered.astype(np.float32) / 255.0 * (vmax - vmin) + vmin

    # Apply the disparity range from calibration
    disparity_filtered[disparity_filtered < vmin] = 0
    disparity_filtered[disparity_filtered > vmax] = 0

    # Calculate the depth map using the filtered disparity map
    depth_map = np.zeros(disparity_filtered.shape, dtype=np.float32)
    valid_disparity_mask = (disparity_filtered > 0)
    depth_map[valid_disparity_mask] = (cam0[0, 0] * baseline_meters) / (disparity_filtered[valid_disparity_mask] + doffs)

    # Invert the depth map if necessary
    depth_map = np.max(depth_map) - depth_map

    # Normalize the depth map for visualization
    depth_map_visual = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_visual = np.uint8(depth_map_visual)

    # Apply a colormap to the depth map for better visualization
    depth_map_colored = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_JET)

    # Save the depth map to the output directory
    output_path_depth = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_depth_map_{filter_name}.png")
    cv2.imwrite(output_path_depth, depth_map_colored)

    # Display the depth map
    cv2.imshow(f'Depth Map ({filter_name} Filtered)', depth_map_colored)
    cv2.waitKey(1)
    return depth_map, output_path_depth

# Calculate and save depth maps for all filtered disparity maps
unfiltered_depth_map, unfiltered_depth_map_path = calculate_and_save_depth_map(disparity_left_8bit, "unfiltered")
median_depth_map, median_depth_map_path = calculate_and_save_depth_map(disparity_left_median_filtered, "median")
bilateral_depth_map, bilateral_depth_map_path = calculate_and_save_depth_map(disparity_left_bilateral_filtered, "bilateral")
wls_depth_map, wls_depth_map_path = calculate_and_save_depth_map(disparity_wls_filtered_8bit, "wls")



def calculate_error_metrics(predicted_depth, ground_truth_depth, epsilon=1e-10):
    # Create a mask for valid ground truth values (greater than zero)
    valid_mask = (ground_truth_depth > epsilon)
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted_depth[valid_mask] - ground_truth_depth[valid_mask]))
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predicted_depth[valid_mask] - ground_truth_depth[valid_mask]) ** 2))
    
    # Calculate MAPE, adding epsilon to prevent division by zero
    mape = np.mean(np.abs((predicted_depth[valid_mask] - ground_truth_depth[valid_mask]) / 
                          (ground_truth_depth[valid_mask] + epsilon))) * 100
    
    # Calculate RMSPE, adding epsilon to prevent division by zero
    rmspe = np.sqrt(np.mean(((predicted_depth[valid_mask] - ground_truth_depth[valid_mask]) / 
                             (ground_truth_depth[valid_mask] + epsilon)) ** 2)) * 100
    
    return mae, rmse, mape, rmspe

# Example usage:
# Resize ground truth depth map to match the predicted depth map dimensions
ground_truth_depth_resized = cv2.resize(ground_truth_depth, 
                                        (unfiltered_depth_map.shape[1], unfiltered_depth_map.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)

# Calculate error metrics for the unfiltered depth map
mae_unfiltered, rmse_unfiltered, mape_unfiltered, rmspe_unfiltered = calculate_error_metrics(unfiltered_depth_map, ground_truth_depth_resized)

# Calculate error metrics for the median filtered depth map
mae_median, rmse_median, mape_median, rmspe_median = calculate_error_metrics(median_depth_map, ground_truth_depth_resized)

# Calculate error metrics for the bilateral filtered depth map
mae_bilateral, rmse_bilateral, mape_bilateral, rmspe_bilateral = calculate_error_metrics(bilateral_depth_map, ground_truth_depth_resized)

# Calculate error metrics for the WLS filtered depth map
mae_wls, rmse_wls, mape_wls, rmspe_wls = calculate_error_metrics(wls_depth_map, ground_truth_depth_resized)

# Save error metrics to a text file
error_metrics_path = os.path.join(output_dir, 'error_metrics.txt')
with open(error_metrics_path, 'w') as f:
    f.write(f"Unfiltered Depth Map - MAE: {mae_unfiltered}\n")
    f.write(f"Median Filtered Depth Map - MAE: {mae_median},")
 
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Disparity map before filtering saved to {output_path_disparity_before}")
print(f"Disparity map after median filtering saved to {output_path_disparity_median}")
print(f"Disparity map after bilateral filtering saved to {output_path_disparity_bilateral}")
print(f"Disparity map after WLS filtering saved to {output_path_disparity_wls}")
print(f"Unfiltered depth map saved to {unfiltered_depth_map_path}")
print(f"Depth map after median filtering saved to {median_depth_map_path}")
print(f"Depth map after bilateral filtering saved to {bilateral_depth_map_path}")
print(f"Depth map after WLS filtering saved to {wls_depth_map_path}")
print(f"Unfiltered Depth Map - MAE: {mae_unfiltered}, RMSE: {rmse_unfiltered}, MAPE: {mape_unfiltered}%, RMSPE: {rmspe_unfiltered}%")
print(f"Median Filtered Depth Map - MAE: {mae_median}, RMSE: {rmse_median}, MAPE: {mape_median}%, RMSPE: {rmspe_median}%")
print(f"Bilateral Filtered Depth Map - MAE: {mae_bilateral}, RMSE: {rmse_bilateral}, MAPE: {mape_bilateral}%, RMSPE: {rmspe_bilateral}%")
print(f"WLS Filtered Depth Map - MAE: {mae_wls}, RMSE: {rmse_wls}, MAPE: {mape_wls}%, RMSPE: {rmspe_wls}%")