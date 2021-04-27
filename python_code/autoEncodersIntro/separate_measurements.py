import pandas as pd
import math
import matplotlib.pyplot as plt


class Separation:
    """A class used for separating ECGs in time windows. Inputs are time_duration in seconds, and file title.
    Proposed time_duration is either 30 seconds for ultra short features, 300 seconds for short features, or
    longer for 24h features"""
    def __init__(self, time_duration: float, file: str):
        # time duration in seconds
        self.time_duration = time_duration
        self.file = file

    def separate_ecgs(self) -> tuple:
        """A method used for separating ECGs. Returns a dictionary of 'data' and 'fs'.
         'data' is a list of pandas.Series objects and 'fs' is the sampling frequency."""
        # example file: "16265.csv"
        data_frame = pd.read_csv(self.file, header=None,
                                 names=['time_step', 'first_measurement', 'second_measurement', 'third_measurement'],
                                 delim_whitespace=True)
        fs = (data_frame.time_step.size - 1) / data_frame.time_step.values[-1]
        # fs = data_frame.time_step[1000] / 1000
        fs = round(fs, 5)
        # print("hello")
        # print(data_frame['time_step'][1])
        result = []
        step = math.ceil(fs*self.time_duration)
        meas = data_frame["first_measurement"]
        for i in range(0, meas.size - 1, step):
            result.append(meas[i:i+step])
        # uncomment below to plot an ecg example of short duration
        # result[0].plot()
        # plt.show()
        return (result, fs)

