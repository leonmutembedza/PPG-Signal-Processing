import os
import csv
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import numpy as np
import heartpy as hp
from collections import deque
import scipy.signal as sp
import pandas as pd
from matplotlib import pyplot as plt
import serial
import peaks as p
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import filters as ft
from globals import bpm_label, raw_data_buffer, timestamps, ser, arduino_port, baud_rate, sampling_rate, fs, amplitude_threshold, window_duration
import mplcursors
from matplotlib.widgets import RectangleSelector


#Recording List
recorded_data = []

# Configure Serial Port  
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Data storage
raw_data_buffer = deque(maxlen=sampling_rate * 10)
timestamps = deque(maxlen=sampling_rate * 10)
start_time = time.time()

# Global state
is_recording = False

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
patient_input.setPlaceholderText("Enter Patient's File Name")
layout.addWidget(patient_input)

# Buttons for Start, Stop, Save, and View Saved Files
button_layout = QtWidgets.QHBoxLayout()

#Button Layout
start_button = QtWidgets.QPushButton("Start Recording")
stop_button = QtWidgets.QPushButton("Stop Recording")
save_button = QtWidgets.QPushButton("Save to CSV File")
view_button = QtWidgets.QPushButton("View Saved Files")
clear_buffer = QtWidgets.QPushButton("Clear Buffer")

#Addition of the buttons to the GUI
button_layout.addWidget(start_button)
button_layout.addWidget(stop_button)
button_layout.addWidget(save_button)
button_layout.addWidget(view_button)
button_layout.addWidget(clear_buffer)
layout.addLayout(button_layout)

#FIltered Data Location
filtered_data_file = r'C:\Users\Katie\source\repos\Pulse Sensor\filtered_data.txt'  
output_csv_file = 'filtered_data.csv'     


'''
                                                                         ______________________________________________________
                                                                        |                                                      |
                                                                        |                Program Functions                     |
                                                                        |                                                      |
                                                                        |      Coded By Leon Mutembedza and Julia Adamus       |
                                                                        |______________________________________________________|
'''
def open_file_viewer():
    global file_window_ref  

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


    def onselect(eclick, erelease, ax):
        """
        This function will handle the rectangle selector, allowing us to
        measure and display the selected region in the plot.
        """
        print(f'Selected area: x1={eclick.xdata}, y1={eclick.ydata} -> x2={erelease.xdata}, y2={erelease.ydata}')
        # You can add your own logic to calculate the area or values within the selection
        ax.set_title(f"Selected region: X1={eclick.xdata:.2f}, Y1={eclick.ydata:.2f} -> X2={erelease.xdata:.2f}, Y2={erelease.ydata:.2f}")
        plt.draw()

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

                # Assuming the CSV has columns "Time" and "PPG"
            time = data['Timestamp'].values  # Time column
            ppg = data['PPG'].values    # PPG signal column

            # Define the sampling frequency
            fs = 100  # Replace with the actual sampling frequency of your data

            # Filter the PPG signal
            lpf_cutoff = 0.7  # Low-pass filter cutoff
            hpf_cutoff = 10   # High-pass filter cutoff
            sos_filter = sp.butter(10, [lpf_cutoff, hpf_cutoff], btype='bp', analog=False, output='sos', fs=fs)
            ppg_filt = sp.sosfiltfilt(sos_filter, ppg)

            # Use HeartPy to detect beats
            working_data, measures = hp.process(ppg_filt, sample_rate=fs)

            # Print HeartPy metrics
            print("HeartPy Metrics:")
            for key, value in measures.items():
                print(f"{key}: {value}")

            # Add interactive cursors to the plot
            mplcursors.cursor(ax, hover=True)

               # Add RectangleSelector for measuring area
            rectangle_selector = RectangleSelector(ax, 
                                                    lambda eclick, erelease: onselect(eclick, erelease, ax),
                                                    useblit=True, button=1, minspanx=5, minspany=5, spancoords='pixels', interactive=True)
            plt.connect('key_press_event', rectangle_selector)
    
            # Plot results with HeartPy
            plt.figure(figsize=(12, 6))
            hp.plotter(working_data, measures)
            plt.title("PPG Signal with Detected Beats")
            plt.show()


            # Keep the window open
            #analysis_window.show()
            analysis_window_ref = analysis_window  # Save reference to prevent garbage collection

        except Exception as e:
            QtWidgets.QMessageBox.critical(file_window, "Error", f"Failed to load or analyze file: {e}")

    # Connect the analyze button
    analyze_button.clicked.connect(analyze_file)

    # Show the file viewer window
    file_window.show()
    file_window_ref = file_window  # Save reference to prevent garbage collection

# Function to save data to a CSV file
def save_to_csv(patient_id):
    if not patient_id:
        bpm_label.setText("Error: Patient ID is missing.")
        return
    try:
        timestampss, ppg_valuess = zip(recorded_data)
        data = pd.DataFrame({"Timestamp": list(timestampss), "PPG": list(ppg_valuess)})
        file_name = f"{patient_id}.csv"
        data.to_csv(file_name, index=False)
        bpm_label.setText(f"Data saved to {file_name}")
    except Exception as e:
        bpm_label.setText(f"Error saving data: {e}")

def transfer_data_to_csv(csv_filename : str , text_filename="recorded_data.txt"):
    data_df.to_csv("filtered.csv", index=None)
    try:
        with open(text_filename, "r") as file:
            rows = file.readlines()

        data = [row.strip().split(",") for row in rows]
        df = pd.DataFrame(data, columns=["Timestamp", "PPG"]) 

        
        df.to_csv(csv_filename + ".csv", index=False)
        bpm_label.setText("Saved Successfully")
        

    except Exception as e:
        bpm_label.setText(f"Error transferring data to CSV: {e}")

def record_data_to_text_file(timestamp, signal, filename="recorded_data.txt"):
    with open(filename, "a") as file:
        file.write(f"{timestamp},{signal}\n") 

def record_data_to_text(timestamp, signal, filename="filtered_data.txt"):
    with open(filename, "a") as file:
        file.write(f"{timestamp},{signal}\n") 

def update():
    global raw_data_buffer, timestamps, data_df

    if not is_recording or patient_id is None:
        return

    try:
        data = ser.readline().decode('utf-8').strip()
        
        if data:
    
            ppg_value = int(data)
            current_time = time.time() - start_time
            raw_data_buffer.append(ppg_value)
            timestamps.append(current_time)
            
            # Filter the data
            filtered_data = ft.butter_highpass_filter(raw_data_buffer, cutoff=0.5, fs=fs)
            filtered_data = ft.butter_lowpass_filter(filtered_data, cutoff=5, fs=fs)
            filtered_data = ft.median_filter(filtered_data, kernel_size=3)

            record_data_to_text_file(current_time, ppg_value)
            record_data_to_text(current_time, ppg_value)

            # Update data_df
            new_data = pd.DataFrame({'Timestamp': [current_time], 'PPG': [filtered_data[-1]]})
            data_df = pd.concat([data_df, new_data], ignore_index=True)

            # Update plot with filtered data
            raw_curve.setData(np.array(timestamps), np.array(filtered_data))

            # Detect peaks and plot them
            peaks, _ = sp.find_peaks(filtered_data)
            peaks_timestamps = np.array(timestamps)[peaks]
            peaks_values = np.array(filtered_data)[peaks]
            peaks_scatter_item.setData(peaks_timestamps, peaks_values)

            # Calculate BPM using HeartPy
            bpm = calculate_bpm_with_heartpy(filtered_data, sampling_rate, window_duration=10)
            bpm_label.setText(f"BPM: {bpm:.2f}")

    except ValueError:
        pass

def read_filtered_data(file_path):
    timestamps = []
    filtered_signals = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                # Check if the line contains a comma
                if ',' in line:
                    timestamp, signal = line.split(',', 1)
                    timestamp = float(timestamp)
                    # Convert signal (array) to a list of floats by splitting space-separated values
                    signal_values = list(map(float, signal.strip('[]').split()))
                    timestamps.append(timestamp)
                    filtered_signals.append(signal_values)
                else:
                    print(f"Skipping invalid line: {line}")  # You can log or handle invalid lines here
    return timestamps, filtered_signals

def write_to_csv(timestamps, filtered_signals, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Timestamp'] + [f'Filtered_Signal_{i+1}' for i in range(len(filtered_signals[0]))])

        # Write data rows
        for timestamp, filtered_signal in zip(timestamps, filtered_signals):
            row = [timestamp] + filtered_signal
            writer.writerow(row)

def save_filtered():
    
    timestamps_filtered, filtered_signals = read_filtered_data(filtered_data_file)
    write_to_csv(timestamps_filtered, filtered_signals, output_csv_file)
    print(f"Data successfully written to {output_csv_file}")

def clear_text_file(filename = "recorded_data.txt"):
    with open(filename, 'w') as file:
        file.truncate(0)

def start_recording():
    global is_recording, start_time, patient_id, data_df
    patient_id = patient_input.text().strip()
    if not patient_id:
        bpm_label.setText("Error: Enter Patient ID")
        return
    raw_data_buffer.clear()
    timestamps.clear()
    ser.reset_input_buffer()
    data_df = pd.DataFrame(columns=["Timestamp", "PPG"])
    is_recording = True
    start_time = time.time()
    clear_text_file()
    bpm_label.setText(f"Recording Started for {patient_id}")

def stop_recording():
    global is_recording
    is_recording = False
    bpm_label.setText("Recording Stopped")
    
def clearr_buffer():
    raw_data_buffer.clear()
    timestamps.clear()
    ser.reset_input_buffer()
    bpm_label.setText("Buffer Cleared")

# Connect buttons
start_button.clicked.connect(start_recording)
stop_button.clicked.connect(stop_recording)
save_button.clicked.connect(lambda: transfer_data_to_csv(patient_input.text().strip()))
view_button.clicked.connect(open_file_viewer)
clear_buffer.clicked.connect(clearr_buffer)



# Timer for updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)


def calculate_bpm_with_heartpy(signal, sample_rate, window_duration=10):

    try:
        # Ensure the signal is within the desired window
        if len(signal) < sample_rate * window_duration:
            #print("Insufficient data for analysis")
            return 0

        # Select the last `window_duration` seconds of data
        windowed_signal = signal[-int(sample_rate * window_duration):]

        # Use HeartPy to process the signal
        working_data, measures = hp.process(windowed_signal, sample_rate=sample_rate)

        # Get the BPM value
        bpm = measures['bpm']

        return bpm
    except Exception as e:
        print(f"HeartPy error: {e}")
        return 0

# Run the application
if __name__ == "__main__":
    win.show()
    QtWidgets.QApplication.instance().exec_()
    open_file_viewer()   