# Enhancing Vehicle Localization Accuracy through GPS-IMU Sensor Fusion Using Particle Filtering

## Overview

This project enhances vehicle localization accuracy by fusing GPS and IMU data using a Particle Filter. The Particle Filter integrates IMU data to predict vehicle states and updates these predictions with GPS measurements to refine localization. This approach aims to improve accuracy over standalone GPS or IMU-based methods.

## Objective

To enhance the accuracy of vehicle localization by fusing GPS and IMU data using Particle Filtering.

## Data Collection

- **IMU Data**: Collect synchronized accelerometer and gyroscope data from the vehicle.
- **GPS Data**: Collect synchronized GPS data including easting, northing, and altitude.

## Preprocessing

1. **Filtering and Cleaning**: Remove noise and outliers from both IMU and GPS data.
2. **Synchronization**: Align timestamps between IMU and GPS data for accurate fusion.

## Algorithm 

### Particle Filter

1. **Prediction Step**: Use IMU data to predict the vehicle's state.
2. **Update Step**: Adjust particle weights based on GPS measurements.

## Fusion Implementation

- **Particle Filter Framework**: Implemented using Python. The filter uses IMU data to predict vehicle states and GPS data to update these predictions.
- **Filtering and Smoothing**: Apply low-pass and high-pass filters to IMU data for noise reduction and use a complementary filter to combine sensor readings.

## Evaluation

- **Metrics**: Compute Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) to evaluate the accuracy of the fused localization compared to standalone GPS and IMU localization.

## Usage

1. **Data Preparation**: Ensure you have the required datasets (`matched_imu_data.csv` and `driving_data_gps.csv`) in the project directory.
2. **Run the Script**: Execute the provided Python script to process data, run the Particle Filter, and visualize results.

```bash
python particle_filter_localization.py
```

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `sklearn`

You can install the dependencies using pip:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```
## Contact
For any questions or comments, please open an issue on this repository or contact Suriya Kasiyalan Siva at k.s.suriya0902@gmail.com
