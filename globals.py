
patient_id = None
patient_input = ""
bpm_label = None
raw_data_buffer = []
timestamps = []
ser = None
arduino_port = "COM3"  
baud_rate = 9600 

#Parameters
sampling_rate = 64
fs = 100
amplitude_threshold = 580
window_duration = 10
buffer_size = 1000
window_size = 100