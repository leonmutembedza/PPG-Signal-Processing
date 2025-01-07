from scipy.signal import butter, filtfilt
# Filter Functions

# Low-Pass Filter
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

# High-Pass Filter
def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

# Band-Pass Filter (for heart rate range)
def butter_bandpass(cutoff_low, cutoff_high, fs, order=4):
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoff_low, cutoff_high, fs, order=4):
    b, a = butter_bandpass(cutoff_low, cutoff_high, fs, order)
    y = filtfilt(b, a, data)
    return y

# Median Filter
from scipy.signal import medfilt

def median_filter(data, kernel_size=3):
    return medfilt(data, kernel_size)
