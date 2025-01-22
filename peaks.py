import numpy as np
import matplotlib.pyplot as plt

def find_peaks_custom(signal, min_height=None, min_distance=1):
    """
    Detect peaks in a PPG signal in real-time.

    Parameters:
    - signal: 1D array-like, the incoming PPG signal data.
    - min_height: Minimum height of peaks.
    - min_distance: Minimum number of samples between neighboring peaks.

    Returns:
    - peaks: Indices of the detected peaks.
    """
    peaks = []
    n = len(signal)

    for i in range(1, n - 1):
        # Check if the current point is a peak
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # Check for minimum height
            if min_height is None or signal[i] >= min_height:
                # Check for minimum distance
                if len(peaks) == 0 or (i - peaks[-1] >= min_distance):
                    peaks.append(i)

    return peaks

