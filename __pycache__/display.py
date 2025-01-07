import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "PPG Signal Analysis Dashboard"

# Define styles for light and dark modes
light_theme = {
    'background': '#FFFFFF',
    'text': '#1E90FF',
    'graph_bg': '#F0F8FF',
    'graph_line': '#1E90FF'
}

dark_theme = {
    'background': '#000000',
    'text': '#D3D3D3',
    'graph_bg': '#2F4F4F',
    'graph_line': '#FFFFFF'
}

# Load data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Layout of the app
app.layout = html.Div([
    dcc.Store(id='theme-store', data='light'),
    dcc.Store(id='selected-file', data=None),

    # Header and Theme Toggle
    html.Div([
        html.H1("PPG Signal Analysis Dashboard", id='title', style={
            "textAlign": "center",
            "fontFamily": "Arial, sans-serif",
            "fontWeight": "bold",
            "fontSize": "2.5em",
            "marginBottom": "10px"
        }),
        html.Button("\u25D2 Toggle Theme", id='toggle-theme', n_clicks=0, style={
            "padding": "10px 20px",
            "fontSize": "1em",
            "color": "#FFFFFF",
            "backgroundColor": "#1E90FF",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer",
            "marginTop": "10px"
        }),
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Left Panel for Folder Selection and File Display
    html.Div([
        html.Label("Select Folder:", style={"fontWeight": "bold", "fontSize": "1.2em"}),
        dcc.Input(id='folder-path', type='text', placeholder='Enter folder path', style={
            "width": "100%",
            "padding": "10px",
            "fontSize": "1em",
            "marginBottom": "10px"
        }),
        html.Button("Load Files", id='load-files', n_clicks=0, style={
            "padding": "10px 20px",
            "fontSize": "1em",
            "color": "#FFFFFF",
            "backgroundColor": "#1E90FF",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer",
            "marginBottom": "10px"
        }),
        html.Div(id='file-list', style={
            "marginTop": "10px",
            "height": "300px",
            "overflowY": "scroll",
            "border": "1px solid #ccc",
            "padding": "10px"
        })
    ], style={"width": "20%", "float": "left", "padding": "10px"}),

    # Main Content Area
    html.Div([
        # Graphs
        html.Div([
            dcc.Graph(id='raw-signal', style={"flex": 1, "margin": "10px"}),
            dcc.Graph(id='processed-signal', style={"flex": 1, "margin": "10px"}),
            dcc.Graph(id='analysis-signal', style={"flex": 1, "margin": "10px"}),
        ], style={"display": "flex", "marginBottom": "20px", "justifyContent": "space-evenly"}),

        # Tools Panel
        html.Div([
            html.Label("Peak Detection Threshold:", style={"fontWeight": "bold", "fontSize": "1.2em"}),
            dcc.Slider(id='peak-threshold', min=0, max=1, step=0.01, value=0.5, tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Lowpass Filter Cutoff (Hz):", style={"fontWeight": "bold", "fontSize": "1.2em", "marginTop": "15px"}),
            dcc.Input(id='lowpass-cutoff', type='number', value=1, step=0.1, style={"padding": "5px", "fontSize": "1em", "border": "1px solid #ccc", "borderRadius": "5px"}),
            html.Label("Fourier Transform Analysis:", style={"fontWeight": "bold", "fontSize": "1.2em", "marginTop": "15px"}),
            html.Button("Apply FFT", id='apply-fft', n_clicks=0, style={
                "padding": "10px 20px",
                "fontSize": "1em",
                "color": "#FFFFFF",
                "backgroundColor": "#1E90FF",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "marginTop": "10px"
            }),
        ], style={"textAlign": "center", "marginBottom": "20px"}),

        # BPM Calculation
        html.Div([
            html.Label("Calculate BPM for Selected Area:", style={"fontWeight": "bold", "fontSize": "1.2em"}),
            html.Button("Calculate BPM", id='calculate-bpm', n_clicks=0, style={
                "padding": "10px 20px",
                "fontSize": "1em",
                "color": "#FFFFFF",
                "backgroundColor": "#1E90FF",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "marginTop": "10px"
            }),
            html.Div(id='bpm-output', style={"marginTop": "10px", "fontSize": "1em", "color": "#1E90FF"}),
        ], style={"textAlign": "center", "marginBottom": "20px"})
    ], style={"width": "75%", "float": "right"}),
], id='main-div', style={"padding": "20px", "overflow": "hidden"})

# Callbacks for interactivity
@app.callback(
    Output('file-list', 'children'),
    [Input('load-files', 'n_clicks')],
    [State('folder-path', 'value')]
)
def display_files(n_clicks, folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return "Invalid folder path."

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        return "No CSV files found in the folder."

    return html.Ul([
        html.Li(f, id={'type': 'file-item', 'index': f}, style={"cursor": "pointer", "color": "#1E90FF"}) for f in files
    ])

@app.callback(
    Output('selected-file', 'data'),
    [Input({'type': 'file-item', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('folder-path', 'value')]
)
def select_file(n_clicks, folder_path):
    ctx = dash.callback_context
    if not ctx.triggered or not folder_path:
        return None

    clicked_file = ctx.triggered[0]['prop_id'].split('.')[0]
    file_name = eval(clicked_file)['index']
    return os.path.join(folder_path, file_name)

@app.callback(
    [Output('raw-signal', 'figure'), Output('processed-signal', 'figure')],
    [Input('selected-file', 'data')]
)
def update_graphs(file_path):
    if not file_path or not os.path.isfile(file_path):
        # Return empty figures if no file is selected
        return go.Figure(), go.Figure()

    # Load data from the selected file
    df = load_data(file_path)
    
    # Check if required columns exist in the DataFrame
    if 'Time' not in df.columns or 'PPG' not in df.columns:
        return go.Figure(), go.Figure()

    # Extract data
    time = df['Time']
    ppg = df['PPG']

    # Raw signal figure
    raw_figure = go.Figure(
        data=[go.Scatter(x=time, y=ppg, mode='lines', line=dict(color='#1E90FF'))],
        layout=go.Layout(
            title='Raw PPG Signal',
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='PPG Amplitude'),
            plot_bgcolor=light_theme['graph_bg'],
        )
    )

    # Processed signal: Apply a lowpass filter
    fs = 100  # Sampling frequency (you can adjust this)
    cutoff = 1  # Default cutoff frequency
    ppg_filtered = butter_lowpass_filter(ppg, cutoff=cutoff, fs=fs)
    processed_figure = go.Figure(
        data=[go.Scatter(x=time, y=ppg_filtered, mode='lines', line=dict(color='#FF4500'))],
        layout=go.Layout(
            title='Processed PPG Signal',
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='PPG Amplitude'),
            plot_bgcolor=light_theme['graph_bg'],
        )
    )

    return raw_figure, processed_figure


if __name__ == '__main__':
    app.run_server(debug=True)
