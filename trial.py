import os
import time
import numpy as np
import pandas as pd
import serial
from collections import deque
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import heartpy as hp
import scipy.signal as sp

# Serial port configuration
arduino_port = 'COM3'  # Change this to your Arduino port
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Data storage
sampling_rate = 100  # Samples per second
raw_data_buffer = deque(maxlen=sampling_rate * 10)
timestamps = deque(maxlen=sampling_rate * 10)

# Initialize Dash app
app = Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Real-Time PPG Signal Monitoring"),
    dcc.Graph(id='ppg-graph'),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0),
    html.Div(id='bpm-output', style={'fontSize': 24}),
    dcc.Dropdown(
        id='filter-dropdown',
        options=[
            {'label': 'Low Pass Filter', 'value': 'lowpass'},
            {'label': 'High Pass Filter', 'value': 'highpass'},
            {'label': 'Band Pass Filter', 'value': 'bandpass'},
        ],
        value='lowpass'
    ),
    html.Button('Start Recording', id='start-button', n_clicks=0),
    html.Button('Stop Recording', id='stop-button', n_clicks=0),
])

# Callback to update the graph and BPM
@app.callback(
    Output('ppg-graph', 'figure'),
    Output('bpm-output', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('start-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
    Input('filter-dropdown', 'value')
)
def update_graph(n_intervals, start_clicks, stop_clicks, filter_type):
    global raw_data_buffer, timestamps

    if start_clicks > stop_clicks:  # If recording is active
        try:
            data = ser.readline().decode('utf-8').strip()
            if data:
                ppg_value = int(data)
                current_time = time.time()
                raw_data_buffer.append(ppg_value)
                timestamps.append(current_time)

                # Convert to numpy arrays for processing
                signal = np.array(raw_data_buffer)
                time_array = np.array(timestamps)

                # Apply filtering based on user selection
                if filter_type == 'lowpass':
                    sos = sp.butter(10, 5, 'low', fs=sampling_rate, output='sos')
                    filtered_signal = sp.sosfiltfilt(sos, signal)
                elif filter_type == 'highpass':
                    sos = sp.butter(10, 0.5, 'high', fs=sampling_rate, output='sos')
                    filtered_signal = sp.sosfiltfilt(sos, signal)
                elif filter_type == 'bandpass':
                    sos = sp.butter(10, [0.5, 5], 'band', fs=sampling_rate, output='sos')
                    filtered_signal = sp.sosfiltfilt(sos, signal)

                # Calculate BPM using HeartPy
                bpm = calculate_bpm_with_heartpy(filtered_signal, sampling_rate)

                # Create the figure
                figure = {
                    'data': [
                        go.Scatter(x=time_array, y=filtered_signal, mode='lines', name='Filtered PPG Signal'),
                    ],
                    'layout' : go.Layout(
                        title='Real-Time PPG Signal',
                        xaxis={'title': 'Time (s)'},
                        yaxis={'title': 'Amplitude'},
                        showlegend=True
                    )
                }

                return figure, f"BPM: {bpm:.2f}"

        except Exception as e:
            return go.Figure(), f"Error: {str(e)}"

    return go.Figure(), "Recording Stopped"

def calculate_bpm_with_heartpy(signal, sample_rate):
    try:
        if len(signal) < sample_rate:
            return 0
        working_data, measures = hp.process(signal, sample_rate=sample_rate)
        return measures['bpm']
    except Exception as e:
        print(f"HeartPy error: {e}")
        return 0

if __name__ == '__main__':
    app.run_server(debug=True)