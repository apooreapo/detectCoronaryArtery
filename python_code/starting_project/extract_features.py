import pandas as pd
import math
# import matplotlib.pyplot as plt
from separate_measurements import Separation
from r_peaks_detection import PeaksDetection
import statistics


ULTRA_SHORT_TIME_WINDOW = 30  # half minute
SHORT_TIME_WINDOW = 300  # 5 minutes


class Extraction:
    """A class for extracting features from ecg. Initialized with file title."""
    def __init__(self, file: str):
        self.file = file
        self.short_measurements = Separation(time_duration=SHORT_TIME_WINDOW, file=self.file)\
            .separate_ecgs()
        self.data = self.short_measurements["data"]
        self.fs = self.short_measurements["fs"]

    def extract_short_features(self):
        """The method for extracting ECG features in an ultra short window."""
        r_peaks = PeaksDetection(fs=self.fs, data_series=self.data[0]).detect_peaks()
        differences = []
        for i in range(1, len(r_peaks)):
            differences.append(r_peaks[i] - r_peaks[i-1])
        normalized_differences = self.__normalized_differences(differences=differences)

        res = self.__sdrr(differences=differences)
        print(f"SDRR is {res} msec.")
        res2 = self.__average_heart_rate(differences=differences)
        print(f"Average heart rate is {res2} bpm.")
        res3 = self.__sdnn(normalized_differences=normalized_differences)
        print(f"SDNN is {res3} msec.")
        res45 = self.__sdann__and_sdnni(data_series=self.data[0])
        res4 = res45["sdann"]
        res5 = res45["sdnni"]
        print(f"SDANN is {res4} msec.")
        print(f"SDNNI is {res5} msec.")

    def extract_ultra_short_features(self):
        """The method for extracting ECG features in an ultra short window."""
        r_peaks = PeaksDetection(fs=self.fs, data_series=self.data[0]).detect_peaks()
        differences = []
        for i in range(1, len(r_peaks)):
            differences.append(r_peaks[i] - r_peaks[i - 1])
        norm = self.__sdrr(differences=differences)
        # print(f"SDRR is {res} msec.")
        res2 = self.__average_heart_rate(differences=differences)
        print(f"Average heart rate is {res2} bpm.")

    def __sdrr(self, differences: list) -> float:
        """Returns the standard deviation of all sinus beats in milliseconds."""
        return statistics.stdev(differences)/self.fs*1000

    def __mean_nn(self, differences: list) -> float:
        """Returns the average duration of a beat in milliseconds."""
        return statistics.mean(differences)/self.fs*1000

    def __sdann__and_sdnni(self, data_series: pd.core.series.Series) -> dict:
        """Returns the standard deviation of the average nn duration of 30s intervals, in milliseconds, and
        the mean of the standard deviation of nn duration of 30s intervals, in milliseconds.
        Dictionary keys are 'sdann' and 'sdnni'."""
        ultra_short_series = self.__extract_ultra_short(data_series=data_series)
        average_nn = []
        nn_std = []
        for ser in ultra_short_series:
            r_peaks = PeaksDetection(fs=self.fs, data_series=ser).detect_peaks()
            differences = []
            for i in range(1, len(r_peaks)):
                differences.append(r_peaks[i] - r_peaks[i - 1])
            normalized_differences = self.__normalized_differences(differences=differences)
            average_nn.append(self.__mean_nn(normalized_differences))
            nn_std.append(statistics.stdev(normalized_differences))
        return {"sdann": statistics.stdev(average_nn), "sdnni": statistics.mean(nn_std)/self.fs*1000}


    def __normalized_differences(self, differences: list) -> list:
        """Returns the time duration of heartbeats, excluding abnormal beats."""
        # step = 1000 / self.fs
        margin = 150 * self.fs / 1000  # maximum margin is 150 msec
        to_be_removed = []
        to_be_removed_indices = []
        for i in range(0, len(differences)):
            to_be_removed.append(False)
        for i in range(0, len(differences) - 1):
            if not to_be_removed[i]:
                for j in [i + 1, i + 2]:
                    if 0 <= j < len(differences):
                        if not to_be_removed[j]:
                            if abs(differences[i] - differences[j]) > margin:
                                temp_sum_i = 0
                                temp_count_i = 0
                                temp_sum_j = 0
                                temp_count_j = 0
                                for k in range(i - 2, i + 2):
                                    if 0 <= k < len(differences):
                                        if not to_be_removed[k]:
                                            temp_sum_i += abs(differences[i]-differences[k])
                                            temp_count_i += 1
                                for k in range(j - 2, j + 2):
                                    if 0 <= k < len(differences):
                                        if not to_be_removed[k]:
                                            temp_sum_j += abs(differences[j]-differences[k])
                                            temp_count_j += 1
                                if temp_sum_i / temp_count_i > temp_sum_j / temp_count_j:
                                    to_be_removed[i] = True
                                    to_be_removed_indices.append(i)
                                else:
                                    to_be_removed[j] = True
                                    to_be_removed_indices.append(j)
        for i in range(len(differences)-1, -1, -1):
            if to_be_removed[i]:
                del differences[i]
        return differences

    def __sdnn(self, normalized_differences: list) -> float:
        """Returns the standard deviation of all normal sinus beats in milliseconds"""
        return statistics.stdev(normalized_differences) / self.fs * 1000

    def __average_heart_rate(self, differences: list) -> float:
        """Return the average heart rate of an ECG in beats per minute"""
        if statistics.mean(differences) != 0:
            return 60/(statistics.mean(differences)/self.fs)
        else:
            return 0
    #     we've got error here

    def __extract_ultra_short(self, data_series: pd.core.series.Series) -> list:
        """Returns ultra short series from a short series"""
        result = []
        step = math.ceil(self.fs * ULTRA_SHORT_TIME_WINDOW)
        meas = data_series
        for i in range(0, meas.size - 1, step):
            result.append(meas[i:i + step])
        return result


