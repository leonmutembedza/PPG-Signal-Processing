PPG Signal Processing and Analysis (Created by Leon Mutembedza and Julia Adamus)
This project focuses on processing and analyzing Photoplethysmogram (PPG) signals to extract meaningful insights, such as heart rate and peak detection. It utilizes Python libraries like NumPy, Pandas, and Matplotlib for data manipulation and visualization, making it a comprehensive tool for studying physiological signals.

Table of Contents
Introduction
Features
Technologies Used
Setup
Usage
Contributing
License
Introduction
Photoplethysmography (PPG) is a simple and non-invasive optical technique used to measure blood volume changes in the microvascular bed of tissue. This project demonstrates how to process raw PPG signals collected from devices like Arduino sensors, including:

Detecting peaks for heart rate calculation.
Handling noisy data through preprocessing.
Visualizing signal trends.
Features
Signal Preprocessing:

Smooth noisy signals using filters (e.g., moving averages).
Normalize PPG data for consistent analysis.
Peak Detection:

Real-time detection of peaks to calculate beats per minute (BPM).
Custom rules for threshold-based peak identification.
Heart Rate Analysis:

Compute heart rate (BPM) dynamically.
Handle outliers by capping BPM values above a threshold.
Visualization:

Plot raw and processed PPG signals.
Annotate peaks and highlight signal trends.
Technologies Used
Python:
Libraries: NumPy, Pandas, Matplotlib, and SciPy.
Hardware:
Arduino sensor for real-time data collection.
Data Visualization:
Interactive plots for signal analysis.
