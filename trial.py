import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time
import serial  # For reading data from serial port

# Set up the serial connection (replace with your actual COM port)
ser = serial.Serial('COM4', 9600)  # Adjust port and baud rate

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Real-Time Data Plotting'),
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='graph-update',
        interval=1000,  # Update every second
        n_intervals=0
    )
])

# Define the callback to update the graph
@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_graph(n):
    # Read data from the serial port (replace with your sensor's data)
    data = ser.readline().decode('utf-8').strip()
    if data:
        ppg_value = int(data)  # Assuming the sensor sends integer values

        # Simulate or get timestamps and values
        x_data = list(range(n+1))  # Timestamps
        y_data = [ppg_value for _ in range(n+1)]  # Real-time PPG values

        # Create the graph figure
        figure = {
            'data': [
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name='PPG Signal'
                )
            ],
            'layout': go.Layout(
                title='Real-Time PPG Data',
                xaxis={'title': 'Time (s)'},
                yaxis={'title': 'PPG Signal'},
                showlegend=True
            )
        }

        return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
