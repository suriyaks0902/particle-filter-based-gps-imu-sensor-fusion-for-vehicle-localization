# Import required libraries
import math
import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt
import matplotlib.animation as animation
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Set plot parameters
plt.rcParams.update({'font.size': 16})

# ----------------------- Data Loading and Processing ------------------------

# Read IMU and GPS data
readings = pd.read_csv('matched_imu_data.csv')
read = pd.read_csv('driving_data_gps.csv')

# Calculate original sampling rates
imu_sampling_rate = 1 / np.mean(np.diff(readings['Time']))
gps_sampling_rate = 1 / np.mean(np.diff(read['Time']))

print(f"IMU Sampling Rate: {imu_sampling_rate:.2f} Hz")
print(f"GPS Sampling Rate: {gps_sampling_rate:.2f} Hz")

# GPS data processing
mov_northing = read['utm_northing'].values
mov_easting = read['utm_easting'].values
mov_altitude = read['altitude'].values
mov_time = read['Time'].values - (read['Time'].values.min())

# Calculate time differences and velocity components
time_diff = np.diff(mov_time)

# Calculate position differentials
easting_diff = np.diff(mov_easting)
northing_diff = np.diff(mov_northing)

# Calculate velocity components (in meters per second)
velocity_east = easting_diff / time_diff
velocity_north = northing_diff / time_diff

# Calculate overall velocity magnitude
velocity_mag = np.sqrt(velocity_east**2 + velocity_north**2)

# Latitude and longitude data
latitude = read['latitude'].values
longitude = read['longitude'].values

# Normalize time for IMU data
readings['Time'] = readings['Time'] - readings['Time'].min()

# ----------------------- IMU Data Normalization ------------------------

# Normalize angular velocity data
readings['IMU.angular_velocity.x'] -= np.mean(readings['IMU.angular_velocity.x'])
readings['IMU.angular_velocity.y'] -= np.mean(readings['IMU.angular_velocity.y'])
readings['IMU.angular_velocity.z'] -= np.mean(readings['IMU.angular_velocity.z'])

# Normalize linear acceleration data
readings['IMU.linear_acceleration.x'] -= np.mean(readings['IMU.linear_acceleration.x'])
readings['IMU.linear_acceleration.y'] -= np.mean(readings['IMU.linear_acceleration.y'])
readings['IMU.linear_acceleration.z'] -= np.mean(readings['IMU.linear_acceleration.z'])

# ----------------------- Interpolation ------------------------

# Interpolate IMU data to GPS timestamps
imu_interpolated = pd.DataFrame({
    'Time': read['Time'],  # Use GPS timestamps as the new time base
    'gyro_x': np.interp(read['Time'], readings['Time'], readings['IMU.angular_velocity.x']),
    'gyro_y': np.interp(read['Time'], readings['Time'], readings['IMU.angular_velocity.y']),
    'gyro_z': np.interp(read['Time'], readings['Time'], readings['IMU.angular_velocity.z']),
    'acc_x': np.interp(read['Time'], readings['Time'], readings['IMU.linear_acceleration.x']),
    'acc_y': np.interp(read['Time'], readings['Time'], readings['IMU.linear_acceleration.y']),
    'acc_z': np.interp(read['Time'], readings['Time'], readings['IMU.linear_acceleration.z'])
})
print("imu_interpolated.head() : ",imu_interpolated.head())

# Calculate new sampling rates after interpolation
new_imu_sampling_rate = 1 / np.mean(np.diff(imu_interpolated['Time']))
new_gps_sampling_rate = 1 / np.mean(np.diff(read['Time']))

print(f"New IMU Sampling Rate: {new_imu_sampling_rate:.2f} Hz")
print(f"New GPS Sampling Rate: {new_gps_sampling_rate:.2f} Hz")

# ----------------------- Data Smoothing and Filtering ------------------------

# Apply moving average filter to GPS data
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 5  # Adjust window size as needed
smoothed_northing = moving_average(mov_northing, window_size)
smoothed_easting = moving_average(mov_easting, window_size)
smoothed_altitude = moving_average(mov_altitude, window_size)

# Update GPS time for smoothed data
smoothed_time = mov_time[:len(smoothed_northing)]

# Detect and remove outliers in GPS data
def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    return data[np.abs(data - mean) < threshold * std]

filtered_northing = remove_outliers(smoothed_northing)
filtered_easting = remove_outliers(smoothed_easting)
filtered_altitude = remove_outliers(smoothed_altitude)

# Calculate yaw angle from GPS
delta_lat = np.diff(latitude)
delta_lon = np.diff(longitude)
yaw_angle = np.degrees(np.arctan2(delta_lat, delta_lon))

# ----------------------- Magnetometer Calibration and Yaw Calculation ------------------------

# Magnetometer heading calculation
magnetometer_heading = np.degrees(np.arctan2(readings['MagField.magnetic_field.x'], readings['MagField.magnetic_field.y']))

# Apply distortion model to magnetometer data
def distortion_model(X_meas, dist_params):
    x = dist_params[0] * X_meas[0] + dist_params[2]
    y = dist_params[1] * X_meas[1] + dist_params[3]
    return np.array([x, y])

x_meas = readings['MagField.magnetic_field.x'].values
y_meas = readings['MagField.magnetic_field.y'].values
X_meas = np.array([x_meas, y_meas])

calibration_params = [8.78849909e-03, 5.81238223e-03, 2.57920090e-06, 1.11092104e-04]
X_model = distortion_model(X_meas, calibration_params)

magnetometer_yaw = np.degrees(np.arctan2(X_model[0], X_model[1]))

# ----------------------- Filtering Functions ------------------------

# Calculate yaw angle from gyro integration
dt = np.mean(np.diff(readings['Time']))

gyro_bias = np.mean(imu_interpolated['gyro_z'])

gyro_yaw_rate = imu_interpolated['gyro_z'] - gyro_bias
integrated_yaw = cumtrapz(gyro_yaw_rate, initial=0) * np.rad2deg(dt)

# Define Butterworth filter functions
def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Apply low-pass and high-pass filters
fs = new_imu_sampling_rate
cutoff_freq_low = 0.1
filtered_magnetometer_yaw = butter_lowpass_filter(magnetometer_yaw, cutoff_freq_low, fs)

cutoff_freq_high = 0.01
filtered_gyro_yaw = butter_highpass_filter(integrated_yaw, cutoff_freq_high, fs)

# Complementary filter
alpha = 0.98  
min_len = min(len(filtered_magnetometer_yaw), len(filtered_gyro_yaw))
complementary_yaw = alpha * filtered_magnetometer_yaw[:min_len] + (1 - alpha) * filtered_gyro_yaw[:min_len]

# ----------------------- Particle Filter Implementation ------------------------

# Particle Filter functions
def transition(particles, dt, acc_x, gyro_z):
    noise = np.random.normal(0, 15, particles.shape)
    particles[:, 0] += particles[:, 3] * dt * np.cos(particles[:, 2]) + noise[:, 0]
    particles[:, 1] += particles[:, 3] * dt * np.sin(particles[:, 2]) + noise[:, 1]
    particles[:, 2] += gyro_z * dt + noise[:, 2]  # Update orientation with gyro_z
    particles[:, 3] += acc_x * dt + noise[:, 3]  # Update velocity with acceleration
    return particles

def update_weights(particles, gps_easting, gps_northing):
    distances = np.sqrt((particles[:, 0] - gps_easting)**2 + (particles[:, 1] - gps_northing)**2)
    weights = np.exp(-distances / 10.0)
    weights /= np.sum(weights)
    return weights

def resample(particles, weights):
    num_particles = len(weights)
    indices = np.arange(num_particles)
    resampled_indices = np.random.choice(indices, size=num_particles, p=weights)
    return particles[resampled_indices]

# Print lengths of data
print(f"Length of imu_interpolated: {len(imu_interpolated)}")
print(f"Length of filtered_easting: {len(filtered_easting)}")
print(f"Length of filtered_northing: {len(filtered_northing)}")

# Initialize Particle Filter around initial GPS position
num_particles = 35000
particles = np.empty((num_particles, 5))
particles[:, 0] = filtered_easting[0] + np.random.normal(0, 1, num_particles)  # Easting
particles[:, 1] = filtered_northing[0] + np.random.normal(0, 1, num_particles) # Northing
particles[:, 2] = np.random.normal(0, 0.1, num_particles)  # Orientation
particles[:, 3] = np.random.normal(0, 1, num_particles)    # Velocity
particles[:, 4] = np.random.normal(0, 0.1, num_particles)  # Angular velocity

# Initialize weights
weights = np.ones(num_particles) / num_particles

# Run Particle Filter
predicted_states_pf = []

# Determine the shortest length among the arrays
min_len = min(len(filtered_easting), len(filtered_northing), len(imu_interpolated))

# for i in range(len(filtered_forward_velocity)):
for i in range(min_len):
    dt = imu_interpolated['Time'].iloc[i+1] - imu_interpolated['Time'].iloc[i]
    # dt = imu_interpolated['Time'].iloc[i+1] - imu_interpolated['Time'].iloc[i] if i + 1 < len(imu_interpolated) else 0
    acc_x = imu_interpolated['acc_x'].iloc[i]
    gyro_z = imu_interpolated['gyro_z'].iloc[i]
    
    # Particle transition with IMU data
    particles = transition(particles, dt, acc_x, gyro_z)
    
    # Update weights based on GPS data
    weights = update_weights(particles, filtered_easting[i], filtered_northing[i])
    
    # Resample particles based on updated weights
    particles = resample(particles, weights)
    
    # Store the mean state of the particles
    predicted_states_pf.append(np.mean(particles, axis=0))

predicted_states_pf = np.array(predicted_states_pf)[:, :2]

# Smooth the particle filter predictions

window_size = 5  
smoothed_states_pf = uniform_filter1d(predicted_states_pf, size=window_size, axis=0)

# ----------------------- Metrics and Validation ------------------------

# Placeholder for IMU positions calculation
def compute_imu_positions(yaw_rate):
    # Integrate yaw rate to get positions, using cumulative sum as a placeholder
    x_positions = np.cumsum(np.cos(yaw_rate))
    y_positions = np.cumsum(np.sin(yaw_rate))
    return np.column_stack((x_positions, y_positions))

# Compute IMU positions
imu_positions = compute_imu_positions(imu_interpolated['gyro_z'])

# Ensure lengths are consistent
min_len = min(len(filtered_easting), len(predicted_states_pf), len(imu_positions))
filtered_easting = filtered_easting[:min_len]
filtered_northing = filtered_northing[:min_len]
predicted_states_pf = predicted_states_pf[:min_len]
imu_positions = imu_positions[:min_len]

# Ensure imu_positions is 2D
if imu_positions.ndim == 1:
    imu_positions = np.column_stack((imu_positions, np.zeros_like(imu_positions))) 

# Compute RMSE and MAE for easting and northing
def compute_metrics(ground_truth_easting, ground_truth_northing, predictions_easting, predictions_northing):
    rmse_easting = np.sqrt(mean_squared_error(ground_truth_easting, predictions_easting))
    rmse_northing = np.sqrt(mean_squared_error(ground_truth_northing, predictions_northing))
    mae_easting = mean_absolute_error(ground_truth_easting, predictions_easting)
    mae_northing = mean_absolute_error(ground_truth_northing, predictions_northing)
    return (rmse_easting, rmse_northing), (mae_easting, mae_northing)

# Compute metrics for Particle Filter
particle_filter_rmse, particle_filter_mae = compute_metrics(filtered_easting, filtered_northing, predicted_states_pf[:, 0], predicted_states_pf[:, 1])
# Uncomment and compute metrics for IMU if available
imu_rmse, imu_mae = compute_metrics(filtered_easting, filtered_northing, imu_positions[:, 0], imu_positions[:, 1])  

print(f"Particle Filter RMSE (Easting): {particle_filter_rmse[0]:.2f} meters")
print(f"Particle Filter RMSE (Northing): {particle_filter_rmse[1]:.2f} meters")
print(f"Particle Filter MAE (Easting): {particle_filter_mae[0]:.2f} meters")
print(f"Particle Filter MAE (Northing): {particle_filter_mae[1]:.2f} meters")

# IMU Error
print(f"IMU RMSE (Easting): {imu_rmse[0]:.2f} meters")
print(f"IMU RMSE (Northing): {imu_rmse[1]:.2f} meters")
print(f"IMU MAE (Easting): {imu_mae[0]:.2f} meters")
print(f"IMU MAE (Northing): {imu_mae[1]:.2f} meters")

# ----------------------- Visualization ------------------------

# Plot Particle Filter Results
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(filtered_easting, filtered_northing, label="Ground Truth GPS Path")
ax.plot(smoothed_states_pf[:, 0], smoothed_states_pf[:, 1], label='Smoothed Particle Filter Estimated Path')
# ax.plot(predicted_states_pf[:, 0], predicted_states_pf[:, 1], label="Particle Filter Estimated Path")
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
plt.title('Particle Filter Prediction')
ax.legend()
plt.show()
