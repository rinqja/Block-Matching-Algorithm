import cv2
import numpy as np
import os

# Camera parameters from calibration.txt
cam0 = np.array([[2826.171, 0, 1292.2],
                 [0, 2826.171, 965.806],
                 [0, 0, 1]])
doffs = 123.77  # Disparity offset in pixels
baseline = 178.089  # Baseline in millimeters
vmin = 34
vmax = 220

# Convert baseline from millimeters to meters for depth calculation
baseline_meters = baseline / 1000.0

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the images
left_image_filename = 'AloeLeft.jpg'
right_image_filename = 'AloeRight.jpg'
ground_truth_filename = 'output_image.png'
left_image_path = os.path.join(script_dir, 'images', left_image_filename)
right_image_path = os.path.join(script_dir, 'images', right_image_filename)
ground_truth_path = os.path.join(script_dir, 'images', ground_truth_filename)

# Debugging: Print the paths
print(f"Loading left image from: {left_image_path}")
print(f"Loading right image from: {right_image_path}")
print(f"Loading ground truth depth map from: {ground_truth_path}")

# Check if files exist
if not os.path.exists(left_image_path):
    print(f"Left image file not found: {left_image_path}")
if not os.path.exists(right_image_path):
    print(f"Right image file not found: {right_image_path}")
if not os.path.exists(ground_truth_path):
    print(f"Ground truth file not found: {ground_truth_path}")

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
block_size = 11

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

# Apply median filtering to the 8-bit disparity map to reduce noise
disparity_left_median_filtered = cv2.medianBlur(disparity_left_8bit, 9)  # Ensure kernel size is odd

# Save and display the median filtered disparity map
output_path_disparity_after_median = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_median_filter.png")
cv2.imwrite(output_path_disparity_after_median, disparity_left_median_filtered)
cv2.imshow('Disparity Map (After Median Filter)', disparity_left_median_filtered)

# Apply bilateral filtering to the median filtered disparity map to reduce noise further while preserving edges
disparity_left_bilateral_filtered = cv2.bilateralFilter(disparity_left_median_filtered, d=9, sigmaColor=75, sigmaSpace=75)

# Save and display the bilateral filtered disparity map
output_path_disparity_after_bilateral = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_disparity_map_bilateral_filter.png")
cv2.imwrite(output_path_disparity_after_bilateral, disparity_left_bilateral_filtered)
cv2.imshow('Disparity Map (After Bilateral Filter)', disparity_left_bilateral_filtered)

# Convert the bilateral filtered disparity map back to the original scale if necessary
disparity_left_final_filtered = disparity_left_bilateral_filtered.astype(np.float32) / 255.0 * (vmax - vmin) + vmin

# Apply the disparity range from calibration
disparity_left_final_filtered[disparity_left_final_filtered < vmin] = 0
disparity_left_final_filtered[disparity_left_final_filtered > vmax] = 0

# Calculate the depth map using the final filtered disparity map
depth_map = np.zeros(disparity_left_final_filtered.shape, dtype=np.float32)
valid_disparity_mask = (disparity_left_final_filtered > 0)
depth_map[valid_disparity_mask] = (cam0[0, 0] * baseline_meters) / (disparity_left_final_filtered[valid_disparity_mask] + doffs)

# Invert the depth map if necessary
depth_map = np.max(depth_map) - depth_map

# Normalize the depth map for visualization
depth_map_visual = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_map_visual = np.uint8(depth_map_visual)

# Apply a colormap to the depth map for better visualization
depth_map_colored = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_JET)

# Save the depth map to the output directory
output_path_depth = os.path.join(output_dir, f"{os.path.splitext(left_image_filename)[0]}_depth_map_final_filtered.png")
cv2.imwrite(output_path_depth, depth_map_colored)

# Display the final depth map
cv2.imshow('Depth Map (Final Filtered)', depth_map_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize ground truth depth map to match the predicted depth map dimensions
ground_truth_depth_resized = cv2.resize(ground_truth_depth, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)

# Function to calculate error metrics
def calculate_error_metrics(predicted_depth, ground_truth_depth):
    valid_mask = (ground_truth_depth > 0)
    mae = np.mean(np.abs(predicted_depth[valid_mask] - ground_truth_depth[valid_mask]))
    rmse = np.sqrt(np.mean((predicted_depth[valid_mask] - ground_truth_depth[valid_mask]) ** 2))
    return mae, rmse

# Calculate error metrics for the final depth map
mae, rmse = calculate_error_metrics(depth_map, ground_truth_depth_resized)

# Save error metrics to a text file
error_metrics_path = os.path.join(output_dir, 'error_metrics.txt')
with open(error_metrics_path, 'w') as f:
    f.write(f"Final Filtered Depth Map - MAE: {mae}, RMSE: {rmse}\n")

# Print results to console
print(f"Disparity map before filtering saved to {output_path_disparity_before}")
print(f"Disparity map after median filtering saved to {output_path_disparity_after_median}")
print(f"Disparity map after bilateral filtering saved to {output_path_disparity_after_bilateral}")
print(f"Final filtered depth map saved to {output_path_depth}")
print(f"Final Filtered Depth Map - MAE: {mae}, RMSE: {rmse}")
