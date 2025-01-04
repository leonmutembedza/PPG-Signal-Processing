import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sqlite3
import time
import numpy as np
from collections import deque
from scipy.signal import find_peaks
import pandas as pd
from matplotlib import pyplot as plt
import serial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# Configure Serial Port
arduino_port = "COM4"  # Replace with your Arduino's port (e.g., COM3 or /dev/ttyUSB0)
baud_rate = 9600   # Match the Arduino's baud rate
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Parameters
sampling_rate = 64
amplitude_threshold = 300
window_duration = 10


# Data storage
raw_data_buffer = deque(maxlen=sampling_rate * 10)
timestamps = deque(maxlen=sampling_rate * 10)
start_time = time.time()

# Global state
is_recording = False
patient_id = None

# PyQtGraph setup
app = QtWidgets.QApplication([])
win = QtWidgets.QMainWindow()
win.setWindowTitle("Real-Time PPG Signal")

# Central widget and layout
central_widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
central_widget.setLayout(layout)
win.setCentralWidget(central_widget)

# Graph widget
plot_widget = pg.GraphicsLayoutWidget()
layout.addWidget(plot_widget)

# Raw Signal Plot
raw_plot = plot_widget.addPlot(title="PPG Signal")
raw_curve = raw_plot.plot(pen='g')

# Peaks Plot
peaks_scatter_item = raw_plot.plot(
    pen=None,
    symbol='o',
    symbolBrush='r'
)

# Label for BPM
bpm_label = pg.LabelItem(justify="right")
plot_widget.addItem(bpm_label)

# Input field for Patient ID
patient_input = QtWidgets.QLineEdit()
patient_input.setPlaceholderText("Enter Patient ID")
layout.addWidget(patient_input)

# Buttons for Start, Stop, Save, and View Saved Files
button_layout = QtWidgets.QHBoxLayout()

start_button = QtWidgets.QPushButton("Start Recording")
stop_button = QtWidgets.QPushButton("Stop Recording")
save_button = QtWidgets.QPushButton("Save to CSV")
view_button = QtWidgets.QPushButton("View Saved Files")
button_layout.addWidget(start_button)
button_layout.addWidget(stop_button)
button_layout.addWidget(save_button)
button_layout.addWidget(view_button)
layout.addLayout(button_layout)



def open_file_viewer():
    global file_window_ref  # Persistent reference to prevent garbage collection

    # Create the file viewer window
    file_window = QtWidgets.QMainWindow()
    file_window.setWindowTitle("Saved Files")
    file_widget = QtWidgets.QWidget()
    file_layout = QtWidgets.QVBoxLayout()
    file_widget.setLayout(file_layout)
    file_window.setCentralWidget(file_widget)

    # List CSV files
    file_list_widget = QtWidgets.QListWidget()
    try:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if not csv_files:
            file_list_widget.addItem("No CSV files found.")
        else:
            file_list_widget.addItems(csv_files)
    except Exception as e:
        QtWidgets.QMessageBox.critical(file_window, "Error", f"Failed to list files: {e}")

    file_layout.addWidget(file_list_widget)

    # Add analyze button
    analyze_button = QtWidgets.QPushButton("Analyze Selected File")
    file_layout.addWidget(analyze_button)

    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )

    analysis_window_ref = None  # Add a global reference to hold the analysis window

    def analyze_file():
        global analysis_window_ref  # Use global to persist the window

        selected_item = file_list_widget.currentItem()
        if not selected_item or selected_item.text() == "No CSV files found.":
            QtWidgets.QMessageBox.warning(file_window, "No File Selected", "Please select a file to analyze.")
            return

        file_path = os.path.join(os.getcwd(), selected_item.text())
        try:
            data = pd.read_csv(file_path)
            if 'Timestamp' not in data.columns or 'PPG' not in data.columns:
                QtWidgets.QMessageBox.critical(file_window, "Invalid File",
                                               "The selected file does not contain required 'Timestamp' or 'PPG' columns.")
                return

            # Create analysis window and keep it persistent
            analysis_window = QtWidgets.QMainWindow()
            analysis_window.setWindowTitle(f"Analysis: {file_path}")
            analysis_widget = QtWidgets.QWidget()
            analysis_layout = QtWidgets.QVBoxLayout()
            analysis_widget.setLayout(analysis_layout)
            analysis_window.setCentralWidget(analysis_widget)

            # Create a Matplotlib figure and canvas
            figure, ax = plt.subplots(figsize=(10, 6))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, analysis_window)  # Add interactive toolbar

            # Add toolbar and canvas to the layout
            analysis_layout.addWidget(toolbar)
            analysis_layout.addWidget(canvas)

            # Plot the PPG Signal
            ax.plot(data['Timestamp'], data['PPG'], label="PPG Signal", color='blue')

            # Detect and plot peaks
            peaks, _ = find_peaks(data['PPG'], height=0.1)
            ax.scatter(data['Timestamp'].iloc[peaks], data['PPG'].iloc[peaks], color='red', label="Detected Peaks")

            # Add labels and title
            ax.set_title(f"PPG Signal from {file_path}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("PPG")
            ax.legend()

            # Render the canvas
            canvas.draw()

            # Keep the window open
            analysis_window.show()
            analysis_window_ref = analysis_window  # Save reference to prevent garbage collection

        except Exception as e:
            QtWidgets.QMessageBox.critical(file_window, "Error", f"Failed to load or analyze file: {e}")

    # Connect the analyze button
    analyze_button.clicked.connect(analyze_file)

    # Show the file viewer window
    file_window.show()
    file_window_ref = file_window  # Save reference to prevent garbage collection


# Connect the "View Files" button to the file viewer function
view_button.clicked.connect(open_file_viewer)


# Function to save data to a CSV file
def save_to_csv(patient_id):
    if not patient_id:
        bpm_label.setText("Error: Patient ID is missing.")
        return
    try:
        data = pd.DataFrame({"Timestamp": list(timestamps), "PPG": list(raw_data_buffer)})
        file_name = f"{patient_id}_ppg.csv"
        data.to_csv(file_name, index=False)
        bpm_label.setText(f"Data saved to {file_name}")
    except Exception as e:
        bpm_label.setText(f"Error saving data: {e}")

# Update function
def update():
    global raw_data_buffer, timestamps

    if not is_recording or patient_id is None:
        return

    try:
        data = ser.readline().decode('utf-8').strip()
        if data:
            ppg_value = int(data)
            current_time = time.time() - start_time

            raw_data_buffer.append(ppg_value)
            timestamps.append(current_time)

            raw_curve.setData(list(timestamps), list(raw_data_buffer))

            peaks, _ = find_peaks(raw_data_buffer, height=amplitude_threshold)
            peaks_timestamps = np.array(timestamps)[peaks]
            peaks_values = np.array(raw_data_buffer)[peaks]
            peaks_scatter_item.setData(peaks_timestamps, peaks_values)

            bpm = calculate_bpm(peaks, timestamps, raw_data_buffer, amplitude_threshold, window_duration)
            bpm_label.setText(f"BPM: {bpm:.2f}")
    except ValueError:
        pass

# Start recording
def start_recording():
    global is_recording, start_time, patient_id
    patient_id = patient_input.text().strip()
    if not patient_id:
        bpm_label.setText("Error: Enter Patient ID")
        return
    is_recording = True
    raw_data_buffer.clear()
    timestamps.clear()
    start_time = time.time()
    bpm_label.setText(f"Recording Started for {patient_id}")

# Stop recording
def stop_recording():
    global is_recording
    is_recording = False
    bpm_label.setText("Recording Stopped")

# Connect buttons
start_button.clicked.connect(start_recording)
stop_button.clicked.connect(stop_recording)
save_button.clicked.connect(lambda: save_to_csv(patient_input.text().strip()))
view_button.clicked.connect(open_file_viewer)

# Timer for updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)

# Helper functions from the second code
def resample_time_series(data, resample_interval=0.01):
    time_min = data['Time'].min()
    time_max = data['Time'].max()
    time_resampled = np.arange(time_min, time_max, resample_interval)
    ppg_resampled = np.interp(time_resampled, data['Time'], data['PPG'])
    return pd.DataFrame({'Time': time_resampled, 'PPG': ppg_resampled})

def remove_noise_fourier(signal, sampling_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_signal = np.fft.fft(signal)
    cutoff_freq = 5
    filter_mask = np.abs(freqs) < cutoff_freq
    return np.fft.ifft(fft_signal * filter_mask).real

def detect_peaks(data, threshold=0.1, min_distance=0.1):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > threshold:
            if len(peaks) == 0 or (i - peaks[-1]) > min_distance:
                peaks.append(i)
    return peaks

def calculate_bpm(peaks, timestamps, signal, amplitude_threshold, window_duration):
  
    # Filter peaks based on amplitude threshold
    valid_peaks = [p for p in peaks if signal[p] > amplitude_threshold]

    if len(valid_peaks) > 1:
        # Get the timestamps of the valid peaks
        peak_times = np.array(timestamps)[valid_peaks]

        # Consider only peaks within the defined time window
        recent_peak_times = peak_times[peak_times > (peak_times[-1] - window_duration)]

        # Calculate time intervals between consecutive peaks
        intervals = np.diff(recent_peak_times)

        if len(intervals) > 0:
            # Calculate BPM (beats per minute)
            bpm = 60 / np.mean(intervals)
            return bpm

    return 0  # Return 0 if not enough peaks are detected

def plot_data_with_peaks(data, peaks, column_name):
    plt.plot(data['Time'], data[column_name], label=f'{column_name} Data', alpha=0.7)
    plt.plot(data['Time'].iloc[peaks], data[column_name].iloc[peaks], "x", label='Detected Peaks', color='red')
    plt.title(f'{column_name} Data with Detected Peaks')
    plt.xlabel("Time (s)")
    plt.ylabel(column_name)
    plt.legend()
    plt.show()

# Run the application
if __name__ == "__main__":
    win.show()
    QtWidgets.QApplication.instance().exec_()
