import pandas as pd
import math
import numpy as np
# import matplotlib.pyplot as plt
from separate_measurements import Separation
from r_peaks_detection import PeaksDetection
import statistics
import fft_transform
import pyhrv.nonlinear
import pyhrv.time_domain
import time


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
        current_data = self.data[10]
        print(len(self.data))
        r_peaks = PeaksDetection(fs=self.fs, data_series=self.data[10]).detect_peaks()
        differences = []
        for i in range(1, len(r_peaks)):
            differences.append(r_peaks[i] - r_peaks[i-1])
        normalized_differences = self.__normalized_differences(differences=differences)

        features_list_dir = []
        names = ["SDRR", "Average Heart Rate", "SDNN", "SDSD", "SDANN", "SDNNI", "pNN50", "RMSSD", "HTI",
                 "HR max - HR min", "VLF Energy", "VLF Energy Percentage", "LF Energy", "LF Energy Percentage",
                 "HF Energy", "HF Energy Percentage", "Poincare sd1", "Poincare sd2", "Poincare ratio",
                 "Poincare Ellipsis Area", "Mean Approximate Entropy", "Approximate Entropy std",
                 "Mean Sample Entropy", "Sample Entropy std", "Short-term Fluctuation", "Long-term Fluctuation"]
        units = ["msec", "bpm", "msec", "msec", "msec", "msec", "NaN", "msec", "NaN", "bpm", "NaN", "NaN", "NaN",
                 "NaN", "NaN", "NaN", "msec", "msec", "NaN", "msec^2", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
        calculation_times_app = [0.00069, 0.00032, 0.00062, 0.00128, 0.13853, 0.13853, 0.00052, 0.00073,
                                 0.00016, 0.00004, 0.00559, 0.00559, 0.00559, 0.00559, 0.00559, 0.00559,
                                 0.04680, 0.04680, 0.04680, 0.04680, 0.74070, 0.74070, 1.31020, 1.31020,
                                 0.07168, 0.07168]
        dict = {"Names": names, "Units": units, "Approximate calculation times": calculation_times_app}
        df = pd.DataFrame(data=dict)
        # print(df)
        # df.to_csv(path_or_buf="./features_list.csv")

        time_durations = []

        tic = time.perf_counter()
        res = self.__sdrr(differences=differences)
        toc = time.perf_counter()
        print(f"SDRR is {round(res, 2)} msec.")
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res2 = self.__average_heart_rate(differences=differences)
        print(f"Average heart rate is {round(res2, 1)} bpm.")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res3 = self.__sdnn(normalized_differences=normalized_differences)
        print(f"SDNN is {round(res3, 2)} msec.")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res3b = self.__sdsd(normalized_differences=normalized_differences)
        print(f"SDSD is {round(res3b, 2)} msec.")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res4, res5 = self.__sdann__and_sdnni(data_series=current_data)
        print(f"SDANN is {round(res4, 2)} msec.")
        print(f"SDNNI is {round(res5, 2)} msec.")
        toc = time.perf_counter()
        time_durations.append((toc - tic) / 2)
        time_durations.append((toc - tic) / 2)

        tic = time.perf_counter()
        res6 = self.__pnn50(normalized_differences=normalized_differences)
        print(f"pNN50 percentage is {round(res6*100,1)}%")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res7 = self.__rmssd(normalized_differences=normalized_differences)
        print(f"RMSSD is {round(res7, 2)} msec.")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res7b = self.__hti(normalized_differences=normalized_differences)
        print(f"HTI is {round(res7b, 3)}")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        res8 = self.__hrmaxmin(differences=differences)
        print(f"HR max - HR min is {round(res8, 1)} bpm.")
        toc = time.perf_counter()
        time_durations.append(toc - tic)

        tic = time.perf_counter()
        fourier_transform = fft_transform.FFTTransform(fs=self.fs, data_series=current_data)
        res9a, res9b = fourier_transform.vlf_band()
        print(f"VLF band energy is {res9a}.")
        print(f"VLF band energy is {round(res9b * 100, 2)}% of the full energy.")
        res10a, res10b = fourier_transform.lf_band()
        print(f"LF band energy is {res10a}.")
        print(f"LF band energy is {round(res10b * 100, 2)}% of the full energy.")
        res11a, res11b = fourier_transform.hf_band()
        print(f"HF band energy is {res11a}.")
        print(f"HF band energy is {round(res11b * 100, 2)}% of the full energy.")
        toc = time.perf_counter()
        for _ in range(0, 6):
            time_durations.append((toc - tic) / 6)

        tic = time.perf_counter()
        res12a, res12b, res12c, res12d = self.__poincare(normalized_differences=normalized_differences)
        print(f"Poincare values - sd1: {round(res12a, 2)} ms, sd2: {round(res12b, 2)} ms, sd ratio: {round(res12c, 2)}"
              f", ellipse area {round(res12d, 2)}, ms^2.")
        toc = time.perf_counter()
        for _ in range(0, 4):
            time_durations.append((toc - tic) / 4)

        tic = time.perf_counter()
        res13a, res13b = self.__approximate_entropy_mean_and_std(data_array=current_data.array, m=2, r=0.04, window=5)
        print(f"Mean approximate entropy is {round(res13a, 4)}.")
        print(f"Standard deviation of approximate entropy is {round(res13b, 4)}.")
        toc = time.perf_counter()
        time_durations.append((toc - tic) / 2)
        time_durations.append((toc - tic) / 2)

        tic = time.perf_counter()
        res14a, res14b = self.__sample_entropy_mean_and_std(data_array=current_data.array, r=0.04, window=5)
        toc = time.perf_counter()
        print(f"Mean sample entropy is {round(res14a, 4)}.")
        print(f"Standard deviation of sample entropy is {round(res14b, 4)}.")
        toc = time.perf_counter()
        time_durations.append((toc - tic) / 2)
        time_durations.append((toc - tic) / 2)

        tic = time.perf_counter()
        res15 = pyhrv.nonlinear.dfa(normalized_differences, show=False)
        res15a = res15["dfa_alpha1"]
        res15b = res15["dfa_alpha2"]
        print(f"Alpha value of the short-term fluctuations is {res15a}.")
        print(f"Alpha value of the long-term fluctuations is {res15b}.")
        toc = time.perf_counter()
        time_durations.append((toc - tic) / 2)
        time_durations.append((toc - tic) / 2)

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

    def __sdann__and_sdnni(self, data_series: pd.core.series.Series) -> tuple:
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
        return statistics.stdev(average_nn), statistics.mean(nn_std)/self.fs*1000


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

    def __pnn50(self, normalized_differences: list) -> float:
        """Returns the percentage of adjacent NN intervals that differ from each other by more than 50 ms"""
        count = 0
        margin = 50 * self.fs / 1000
        for i in range(0, len(normalized_differences) - 1):
            if abs(normalized_differences[i] - normalized_differences[i + 1]) > margin:
                count += 1
        sz = len(normalized_differences)
        if sz > 1:
            return count / (sz - 1)
        else:
            return 0

    def __sdsd(self, normalized_differences: list) -> float:
        """Returns the standard deviation of succesive differences of beats in milliseconds"""
        diff = []
        for i in range(0, len(normalized_differences) - 1):
            diff.append(abs(normalized_differences[i] - normalized_differences[i + 1]))
        diff = diff / self.fs * 1000
        return statistics.stdev(diff)


    def __rmssd(self, normalized_differences):
        """Returns RMSSD metric, by using nn intervals (computed in milliseconds)."""
        sum = 0
        normalized_differences_sec = []
        for diff in normalized_differences:
            normalized_differences_sec.append(diff / self.fs)
        for i in range(0, len(normalized_differences_sec) - 1):
            sum += (normalized_differences_sec[i + 1] - normalized_differences_sec[i]) ** 2
        sz = len(normalized_differences_sec)
        if sz > 1:
            return math.sqrt(sum / (sz - 1)) * 1000
        else:
            return 0

    def __hrmaxmin(self, differences):
        """Returns the difference between the max and min heartbeat, in beats per minute."""
        min_diff = 100000
        max_diff = 0
        for i in range(0, len(differences)):
            if differences[i] > max_diff:
                max_diff = differences[i]
            if differences[i] < min_diff:
                min_diff = differences[i]
        max = 60/(max_diff/self.fs)
        min = 60/(min_diff/self.fs)
        return min - max


    def __extract_ultra_short(self, data_series: pd.core.series.Series) -> list:
        """Returns ultra short series from a short series"""
        result = []
        step = math.ceil(self.fs * ULTRA_SHORT_TIME_WINDOW)
        meas = data_series
        for i in range(0, meas.size - 1, step):
            result.append(meas[i:i + step])
        return result

    def __hti(self, normalized_differences) -> float:
        """Calculates and returns Heart Rate Variability Triangular Index.
        We separate the ECG in 8 msec parts, and create the following histogram of nn duration (in 8 msec pieces).
        HTI is equal to the height of the max histogram bar, divided by the count of intervals (i.e. beats)."""
        def count_elements(seq, step) -> dict:
            """Tally elements from `seq`."""
            hist = {}
            for i in seq:
                ind = int(i / step)
                hist[ind] = hist.get(int(ind), 0) + 1
            return hist

        if self.fs <= 125:
            step = 1
        else:
            step = int(8 / (1000 / self.fs))
        hist = count_elements(seq=normalized_differences, step=step)
        hist_list = list(hist)
        temp_max = 0
        for val in hist_list:
            if hist[val] > temp_max:
                temp_max = hist[val]
        return temp_max / len(normalized_differences)

    def __poincare(self, normalized_differences):
        """Returns poincare metrics, as tuple (sd1, sd2, sd ratio, ellipse area)."""
        difs = normalized_differences / self.fs
        res = pyhrv.nonlinear.poincare(nni=difs, show=False)
        return res["sd1"], res["sd2"], res["sd_ratio"], res["ellipse_area"]
        # sd1 is equal to (0.5 * rmssd) ^ 0.5

    def __approximate_entropy_mean_and_std(self, data_array, m, r, window: int) -> tuple:
        """Returns the mean and standard deviation of approximate energy, in a tuple."""

        def approximate_entropy(data_arr, m2, r1):
            data_arr = np.array(data_arr)
            n = data_arr.shape[0]

            def _phi(m1):
                z = n - m1 + 1.0
                x = np.array([data_arr[ii:ii + m1] for ii in range(int(z))])
                x2 = np.repeat(x[:, np.newaxis], 1, axis=2)
                c = np.sum(np.absolute(x - x2).max(axis=2) <= r1, axis=0) / z
                return np.log(c).sum() / z

            return abs(_phi(m2 + 1) - _phi(m2))

        window_samples = int(window * self.fs)
        approximate_energy = []
        for i in range(0, len(data_array), window_samples):
            if i+window_samples < len(data_array):
                approximate_energy.append(approximate_entropy(data_arr=data_array[i:i+window_samples], m2=m, r1=r))
            else:
                approximate_energy.append(approximate_entropy(data_arr=data_array[i:], m2=m, r1=r))
        mn = statistics.mean(approximate_energy)
        sd = statistics.stdev(approximate_energy)
        return mn, sd

    def __sample_entropy_mean_and_std(self, data_array, r, window: int):
        """Returns the mean and standard deviation of sample energy, in a tuple."""

        def sampen(data, m, r1):
            n = len(data)
            b = 0.0
            a = 0.0

            # Split time series and save all templates of length m
            xmi = np.array([data[i: i + m] for i in range(n - m)])
            xmj = np.array([data[i: i + m] for i in range(n - m + 1)])

            # Save all matches minus the self-match, compute B
            b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r1) - 1 for xmii in xmi])

            # Similar for computing A
            m += 1
            xm = np.array([data[i: i + m] for i in range(n - m + 1)])

            a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r1) - 1 for xmi in xm])

            # Return SampEn
            return -np.log(a / b)

        window_samples = int(window * self.fs)
        sample_entropy = []
        for i in range(0, len(data_array), window_samples):
            if i + window_samples < len(data_array):
                sample_entropy.append(sampen(data=data_array[i:i + window_samples], r1=r, m=2))
            else:
                sample_entropy.append(sampen(data=data_array[i:], r1=r, m=2))
        mn = statistics.mean(sample_entropy)
        sd = statistics.stdev(sample_entropy)
        return mn, sd


