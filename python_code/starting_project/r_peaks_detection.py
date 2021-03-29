import matplotlib.pyplot as plt
from ecgdetectors import Detectors
import pandas as pd
import numpy as np
from pan_tompkins import PanTompkins


class PeaksDetection:
    """Class for detecting r peaks in ECGs. Input is a pandas.Series object with the ECG's measurements.
    Output is a numPy array with peaks locations"""

    def __init__(self, fs: float, data_series: pd.core.series.Series):
        self.fs = fs
        self.data_series = data_series

    def detect_peaks(self) -> list:
        """A method to detect R peaks in a pandas data_series."""
        # unfiltered_ecg = self.data_series.to_numpy()
        # detectors = Detectors(self.fs)
        my_pan_tompkins = PanTompkins(fs=self.fs, data_series=self.data_series, print_message=False)
        r_peaks = my_pan_tompkins.pan_tompkins()

        # choose between available r peaks detectors
        # r_peaks = detectors.two_average_detector(unfiltered_ecg)
        # r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
        # r_peaks = detectors.swt_detector(unfiltered_ecg)
        # r_peaks = detectors.engzee_detector(unfiltered_ecg)
        # r_peaks = detectors.christov_detector(unfiltered_ecg)
        # r_peaks = detectors.hamilton_detector(unfiltered_ecg)
        # r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)

        # uncomment below to plot the result
        # plt.figure()
        # plt.plot(unfiltered_ecg)
        # plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
        # plt.title('Detected R-peaks')
        # plt.show()

        return r_peaks
