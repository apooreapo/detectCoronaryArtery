import matplotlib.pyplot as plt
from ecgdetectors import Detectors
import pandas as pd


class SimpleExample:
    """Example class demonstrating how to use r peaks detection in ecg"""
    file = "./example.csv"
    df = pd.read_csv(file)
    unfiltered_ecg = df["first_measurement"].to_numpy()
    print(unfiltered_ecg)
    fs = 128

    detectors = Detectors(fs)

    # r_peaks = detectors.two_average_detector(unfiltered_ecg)
    # r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
    # r_peaks = detectors.swt_detector(unfiltered_ecg)
    r_peaks = detectors.engzee_detector(unfiltered_ecg)
    # r_peaks = detectors.christov_detector(unfiltered_ecg)
    # r_peaks = detectors.hamilton_detector(unfiltered_ecg)
    # r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)

    plt.figure()
    plt.plot(unfiltered_ecg)
    plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
    plt.title('Detected R-peaks')

    plt.show()
