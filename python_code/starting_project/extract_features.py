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
from tqdm import tqdm
import gc
import os
import psutil
import sys


ULTRA_SHORT_TIME_WINDOW = 30  # half minute
SHORT_TIME_WINDOW = 300  # 5 minutes


class Queue:
    """A class implementic a queue. Methods: insert."""
    def __init__(self, size):
        self.size = size
        self.data = []

    def insert_element(self, element):
        data_size = len(self.data)
        if data_size == 0:
            self.data.append(element)
        elif data_size < self.size:
            self.data.append(self.data[-1])
            for i in range(len(self.data) - 2, 0, -1):
                self.data[i] = self.data[i-1]
            self.data[0] = element

        else:
            for i in range(data_size - 1, 0, -1):
                self.data[i] = self.data[i - 1]
            self.data[0] = element


class Extraction:
    """A class for extracting features from ecg. Initialized with file title."""
    def __init__(self, file: str):
        self.file = file
        my_separation = Separation(time_duration=SHORT_TIME_WINDOW, file=self.file)\
            .separate_ecgs()
        my_ultra_short_separation = Separation(time_duration=ULTRA_SHORT_TIME_WINDOW, file=self.file)\
            .separate_ecgs()
        self.short_measurements = my_separation
        self.ultra_short_measurements = my_ultra_short_separation
        self.data = self.short_measurements["data"]
        self.ultra_short_data = self.ultra_short_measurements["data"]
        self.fs = self.short_measurements["fs"]

        self.differences = []
        self.normalized_differences = []
        self.r_peaks = []
        self.current_data = []
        self.factor = int(round(self.fs / 64))

    def extract_short_features(self, print_message=False) -> dict:
        """The method for extracting ECG features in a short window.
        Return a dictionary with the extracted features of the file"""

        # Construct the features_list.csv with the extracted features characteristics.

        # features_list_dir = []
        names = ["SDRR", "Average Heart Rate", "SDNN", "SDSD", "SDANN", "SDNNI", "pNN50", "RMSSD", "HTI",
                 "HR max - HR min", "VLF Energy", "VLF Energy Percentage", "LF Energy", "LF Energy Percentage",
                 "HF Energy", "HF Energy Percentage", "Poincare sd1", "Poincare sd2", "Poincare ratio",
                 "Poincare Ellipsis Area", "Mean Approximate Entropy", "Approximate Entropy std",
                 "Mean Sample Entropy", "Sample Entropy std",
                 "File title", "Has CAD",
                 "VLF Peak", "LF Peak", "HF Peak", "LF/HF Energy"]
        # names = ["VLF Peak", "LF Peak", "HF Peak", "LF/HF Energy"]
        # units = ["msec", "bpm", "msec", "msec", "msec", "msec", "NaN", "msec", "NaN", "bpm", "NaN", "NaN", "NaN",
        #          "NaN", "NaN", "NaN", "msec", "msec", "NaN", "msec^2", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
        # calculation_times_app = [0.00069, 0.00032, 0.00062, 0.00128, 0.13853, 0.13853, 0.00052, 0.00073,
        #                          0.00016, 0.00004, 0.00559, 0.00559, 0.00559, 0.00559, 0.00559, 0.00559,
        #                          0.04680, 0.04680, 0.04680, 0.04680, 0.74070, 0.74070, 1.31020, 1.31020,
        #                          0.07168, 0.07168]
        # feature_dict = {"Names": names, "Units": units, "Approximate calculation times": calculation_times_app}
        # df = pd.DataFrame(data=feature_dict)
        # print(df)
        # df.to_csv(path_or_buf="./features_list.csv")

        tic = time.perf_counter()

        column_titles = names
        # column_titles.append("File Title")
        # column_titles.append("Has CAD")

        res_dict = {}
        for name in column_titles:
            res_dict[name] = []

        # data_length = 1  # use this with print_message = true for testing purposes

        data_length = len(self.data)
        pbar = tqdm(total=data_length)

        for index in range(0, data_length):
            current_data = self.data[index]
            r_peaks = PeaksDetection(fs=self.fs, data_series=current_data).detect_peaks()
            differences = []
            for i in range(1, len(r_peaks)):
                differences.append(r_peaks[i] - r_peaks[i-1])
            normalized_differences = self.__normalized_differences(differences=differences)
            normalized_current_data = self.__normalize_and_downscale(data_series=current_data, factor=self.factor)
            if len(normalized_differences) > 2:
                # time_durations = []

                tic = time.perf_counter()
                res = self.__sdrr(differences=differences)
                res_dict[column_titles[0]].append(res)

                # toc = time.perf_counter()
                if print_message:
                    print(f"SDRR is {round(res, 2)} msec.")
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res2 = self.__average_heart_rate(differences=differences)
                res_dict[column_titles[1]].append(res2)
                if print_message:
                    print(f"Average heart rate is {round(res2, 1)} bpm.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res3 = self.__sdnn(normalized_differences=normalized_differences)
                res_dict[column_titles[2]].append(res3)
                if print_message:
                    print(f"SDNN is {round(res3, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res3b = self.__sdsd(normalized_differences=normalized_differences)
                res_dict[column_titles[3]].append(res3b)
                if print_message:
                    print(f"SDSD is {round(res3b, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res4, res5 = self.__sdann__and_sdnni(data_series=current_data)
                res_dict[column_titles[4]].append(res4)
                res_dict[column_titles[5]].append(res5)
                if print_message:
                    print(f"SDANN is {round(res4, 2)} msec.")
                    print(f"SDNNI is {round(res5, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)

                # tic = time.perf_counter()
                res6 = self.__pnn50(normalized_differences=normalized_differences)
                res_dict[column_titles[6]].append(res6)
                if print_message:
                    print(f"pNN50 percentage is {round(res6*100,1)}%")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res7 = self.__rmssd(normalized_differences=normalized_differences)
                res_dict[column_titles[7]].append(res7)
                if print_message:
                    print(f"RMSSD is {round(res7, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res7b = self.__hti(normalized_differences=normalized_differences)
                res_dict[column_titles[8]].append(res7b)
                if print_message:
                    print(f"HTI is {round(res7b, 3)}")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res8 = self.__hrmaxmin(normalized_differences=normalized_differences)
                res_dict[column_titles[9]].append(res8)
                if print_message:
                    print(f"HR max - HR min is {round(res8, 1)} bpm.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                fourier_transform = fft_transform.FFTTransform(fs=self.fs, data_series=current_data)
                res9a, res9b, res28 = fourier_transform.vlf_band()
                res_dict[column_titles[10]].append(res9a)
                res_dict[column_titles[11]].append(res9b)
                if print_message:
                    print(f"VLF band energy is {res9a}.")
                    print(f"VLF band energy is {round(res9b * 100, 2)}% of the full energy.")
                    print(f"VLF Energy peak is at {round(res28, 3)} Hz.")
                res10a, res10b, res29 = fourier_transform.lf_band()
                res_dict[column_titles[12]].append(res10a)
                res_dict[column_titles[13]].append(res10b)
                if print_message:
                    print(f"LF band energy is {res10a}.")
                    print(f"LF band energy is {round(res10b * 100, 2)}% of the full energy.")
                    print(f"LF Energy peak is at {round(res29, 3)} Hz.")
                res11a, res11b, res30 = fourier_transform.hf_band()
                res_dict[column_titles[14]].append(res11a)
                res_dict[column_titles[15]].append(res11b)
                if print_message:
                    print(f"HF band energy is {res11a}.")
                    print(f"HF band energy is {round(res11b * 100, 2)}% of the full energy.")
                    print(f"HF Energy peak is at {round(res30, 3)} Hz.")
                if res10a > 0:
                    res31 = res10a / res11a
                if print_message:
                    print(f"LF / HF energy is {round(res31, 5)}.")
                # toc = time.perf_counter()
                # for _ in range(0, 6):
                #     time_durations.append((toc - tic) / 6)

                # tic = time.perf_counter()
                res12a, res12b, res12c, res12d = self.__poincare(normalized_differences=normalized_differences)
                res_dict[column_titles[16]].append(res12a)
                res_dict[column_titles[17]].append(res12b)
                res_dict[column_titles[18]].append(res12c)
                res_dict[column_titles[19]].append(res12d)
                if print_message:
                    print(f"Poincare values - sd1: {round(res12a, 2)} ms, sd2: {round(res12b, 2)} ms, sd ratio:"
                          f" {round(res12c, 2)}"
                          f", ellipse area {round(res12d, 2)}, ms^2.")
                # toc = time.perf_counter()
                # for _ in range(0, 4):
                #     time_durations.append((toc - tic) / 4)

                # tic = time.perf_counter()
                res13a, res13b = self.__approximate_entropy_mean_and_std(data_array=normalized_current_data, m=2,
                                                                         r=0.04, window=5)
                res_dict[column_titles[20]].append(res13a)
                res_dict[column_titles[21]].append(res13b)
                if print_message:
                    print(f"Mean approximate entropy is {round(res13a, 4)}.")
                    print(f"Standard deviation of approximate entropy is {round(res13b, 4)}.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)

                # tic = time.perf_counter()
                res14a, res14b = self.__sample_entropy_mean_and_std(data_array=normalized_current_data, r=0.04, window=5)
                res_dict[column_titles[22]].append(res14a)
                res_dict[column_titles[23]].append(res14b)
                # toc = time.perf_counter()
                if print_message:
                    print(f"Mean sample entropy is {round(res14a, 4)}.")
                    print(f"Standard deviation of sample entropy is {round(res14b, 4)}.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)

                # tic = time.perf_counter()
                # res15 = pyhrv.nonlinear.dfa(normalized_differences, show=False)
                # res15a = res15["dfa_alpha1"]
                # res15b = res15["dfa_alpha2"]
                # res_dict[column_titles[24]].append(res15a)
                # res_dict[column_titles[25]].append(res15b)
                # if print_message:
                #     print(f"Alpha value of the short-term fluctuations is {res15a}.")
                #     print(f"Alpha value of the long-term fluctuations is {res15b}.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)
                txt = self.file.split(sep='/')[3]
                res_dict[column_titles[24]].append(txt)
                res_dict[column_titles[25]].append(1)
                res_dict[column_titles[26]].append(res28)
                res_dict[column_titles[27]].append(res29)
                res_dict[column_titles[28]].append(res30)
                res_dict[column_titles[29]].append(res31)
                # print(res_dict)
            else:
                for i in range(0, len(column_titles)):
                    res_dict[column_titles[i]].append(float("nan"))
            pbar.update(1)
            # print(normalized_differences)
            # print(differences)
        pbar.close()
        toc = time.perf_counter()
        print(f"Total extraction time: {round(toc - tic, 3)} sec.")
        # print(res_dict)
        return res_dict

    def extract_ultra_short_features(self, print_message=False):
        """The method for extracting ECG features in an ultra short window."""

        # Construct the features_list.csv with the extracted features characteristics.

        # features_list_dir = []
        names = ["SDRR", "Average Heart Rate", "SDNN", "SDSD", "pNN50", "RMSSD", "HTI",
                 "HR max - HR min", "LF Energy", "LF Energy Percentage",
                 "HF Energy", "HF Energy Percentage", "Poincare sd1", "Poincare sd2", "Poincare ratio",
                 "Poincare Ellipsis Area", "Mean Approximate Entropy", "Standard Deviation of Approximate Entropy",
                 "Mean Sample Entropy", "Standard Deviation of Sample Entropy",
                 "File title", "Has CAD",
                 "LF Peak", "HF Peak", "LF/HF Energy"]

        tic = time.perf_counter()

        column_titles = names
        # column_titles.append("File Title")
        # column_titles.append("Has CAD")

        res_dict = {}
        for name in column_titles:
            res_dict[name] = []

        # data_length = 1  # use this with print_message = true for testing purposes

        data_length = len(self.ultra_short_data)
        # data_length = 10
        pbar = tqdm(total=data_length)

        for index in range(0, data_length):
            self.current_data = self.ultra_short_data[index]
            self.r_peaks = PeaksDetection(fs=self.fs, data_series=self.current_data).detect_peaks()
            self.differences = []
            self.normalized_differences = []
            for i in range(1, len(self.r_peaks)):
                self.differences.append(self.r_peaks[i] - self.r_peaks[i - 1])
            self.normalized_differences = self.__normalized_differences(differences=self.differences)
            normalized_current_data = self.__normalize_and_downscale(data_series=self.current_data, factor=self.factor)
            if len(self.normalized_differences) > 2:
                # time_durations = []

                tic = time.perf_counter()
                res = self.__sdrr(differences=self.differences)
                res_dict[column_titles[0]].append(res)

                # toc = time.perf_counter()
                if print_message:
                    print(f"SDRR is {round(res, 2)} msec.")
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res2 = self.__average_heart_rate(differences=self.differences)
                res_dict[column_titles[1]].append(res2)
                if print_message:
                    print(f"Average heart rate is {round(res2, 1)} bpm.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res3 = self.__sdnn(normalized_differences=self.normalized_differences)
                res_dict[column_titles[2]].append(res3)
                if print_message:
                    print(f"SDNN is {round(res3, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res3b = self.__sdsd(normalized_differences=self.normalized_differences)
                res_dict[column_titles[3]].append(res3b)
                if print_message:
                    print(f"SDSD is {round(res3b, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res4 = self.__pnn50(normalized_differences=self.normalized_differences)
                res_dict[column_titles[4]].append(res4)
                if print_message:
                    print(f"pNN50 percentage is {round(res4 * 100, 1)}%")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res5 = self.__rmssd(normalized_differences=self.normalized_differences)
                res_dict[column_titles[5]].append(res5)
                if print_message:
                    print(f"RMSSD is {round(res5, 2)} msec.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res7b = self.__hti(normalized_differences=self.normalized_differences)
                res_dict[column_titles[6]].append(res7b)
                if print_message:
                    print(f"HTI is {round(res7b, 3)}")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                res8 = self.__hrmaxmin(normalized_differences=self.normalized_differences)
                res_dict[column_titles[7]].append(res8)
                if print_message:
                    print(f"HR max - HR min is {round(res8, 1)} bpm.")
                # toc = time.perf_counter()
                # time_durations.append(toc - tic)

                # tic = time.perf_counter()
                fourier_transform = fft_transform.FFTTransform(fs=self.fs, data_series=self.current_data)
                res10a, res10b, res29 = fourier_transform.lf_band()
                res_dict[column_titles[8]].append(res10a)
                res_dict[column_titles[9]].append(res10b)
                if print_message:
                    print(f"LF band energy is {res10a}.")
                    print(f"LF band energy is {round(res10b * 100, 2)}% of the full energy.")
                    print(f"LF Energy peak is at {round(res29, 3)} Hz.")
                res11a, res11b, res30 = fourier_transform.hf_band()
                res_dict[column_titles[10]].append(res11a)
                res_dict[column_titles[11]].append(res11b)
                if print_message:
                    print(f"HF band energy is {res11a}.")
                    print(f"HF band energy is {round(res11b * 100, 2)}% of the full energy.")
                    print(f"HF Energy peak is at {round(res30, 3)} Hz.")
                if res10a > 0:
                    res31 = res10a / res11a
                else:
                    res31 = 0
                if print_message:
                    print(f"LF / HF energy is {round(res31, 5)}.")
                # toc = time.perf_counter()
                # for _ in range(0, 6):
                #     time_durations.append((toc - tic) / 6)

                # tic = time.perf_counter()
                # res12a = 0
                # res12b = 0
                # res12c = 0
                # res12d = 0
                res12a, res12b, res12c, res12d = self.__poincare(normalized_differences=self.normalized_differences)
                res_dict[column_titles[12]].append(res12a)
                res_dict[column_titles[13]].append(res12b)
                res_dict[column_titles[14]].append(res12c)
                res_dict[column_titles[15]].append(res12d)
                if print_message:
                    print(f"Poincare values - sd1: {round(res12a, 2)} ms, sd2: {round(res12b, 2)} ms, sd ratio:"
                          f" {round(res12c, 2)}"
                          f", ellipse area {round(res12d, 2)}, ms^2.")
                # toc = time.perf_counter()
                # for _ in range(0, 4):
                #     time_durations.append((toc - tic) / 4)

                # tic = time.perf_counter()
                res13a, res13b = self.__approximate_entropy_mean_and_std(data_array=normalized_current_data,
                                                                         m=2,
                                                                         r=0.04, window=5)
                res_dict[column_titles[16]].append(res13a)
                res_dict[column_titles[17]].append(res13b)
                if print_message:
                    print(f"Mean approximate entropy is {round(res13a, 4)}.")
                    print(f"Standard deviation of approximate entropy is {round(res13b, 4)}.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)

                # tic = time.perf_counter()
                res14a, res14b = self.__sample_entropy_mean_and_std(data_array=normalized_current_data,
                                                                    r=0.04, window=5)
                res_dict[column_titles[18]].append(res14a)
                res_dict[column_titles[19]].append(res14b)
                # toc = time.perf_counter()
                if print_message:
                    print(f"Mean sample entropy is {round(res14a, 4)}.")
                    print(f"Standard deviation of sample entropy is {round(res14b, 4)}.")
                # toc = time.perf_counter()
                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)

                # time_durations.append((toc - tic) / 2)
                # time_durations.append((toc - tic) / 2)
                txt = self.file.split(sep='/')[3]
                res_dict[column_titles[20]].append(txt)
                res_dict[column_titles[21]].append(0)
                res_dict[column_titles[22]].append(res29)
                res_dict[column_titles[23]].append(res30)
                res_dict[column_titles[24]].append(res31)
                # print(res_dict)
                del res29, txt, res14a, res14b, res13a, res13b, res12a, res12b, res12c, res12d, res31
                del res30, res11a, res11b, res10a, res10b, res7b, res5, res4, res3b, res2, res

                gc.collect()
            else:
                for i in range(0, len(column_titles)):
                    res_dict[column_titles[i]].append(float("nan"))
            pbar.update(1)
            # print(normalized_differences)
            # print(differences)
        pbar.close()
        toc = time.perf_counter()
        print(f"Total extraction time: {round(toc - tic, 3)} sec.")
        return res_dict
    def __sdrr(self, differences: list) -> float:
        """Returns the standard deviation of all sinus beats in milliseconds."""
        return statistics.stdev(differences)/self.fs*1000

    def __mean_nn(self, differences: list) -> float:
        """Returns the average duration of a beat in milliseconds."""
        return statistics.mean(differences)/self.fs*1000

    def __sdann__and_sdnni(self, data_series) -> tuple:
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
            if len(normalized_differences) > 2:
                average_nn.append(self.__mean_nn(normalized_differences))
                nn_std.append(statistics.stdev(normalized_differences))
        if len(average_nn) > 1 and len(nn_std) > 1:
            return statistics.stdev(average_nn), statistics.mean(nn_std)/self.fs*1000
        else:
            return float("nan"), float("nan")

    def __normalized_differences(self, differences: list) -> list:
        """Returns the time duration of heartbeats, excluding abnormal beats."""
        # step = 1000 / self.fs
        # margin = 150 * self.fs / 1000  # maximum margin is 150 msec
        # to_be_removed = []
        # to_be_removed_indices = []
        # for i in range(0, len(differences)):
        #     to_be_removed.append(False)
        # for i in range(0, len(differences) - 1):
        #     if not to_be_removed[i]:
        #         for j in [i + 1, i + 2]:
        #             if 0 <= j < len(differences):
        #                 if not to_be_removed[j]:
        #                     if abs(differences[i] - differences[j]) > margin:
        #                         temp_sum_i = 0
        #                         temp_count_i = 0
        #                         temp_sum_j = 0
        #                         temp_count_j = 0
        #                         for k in range(i - 2, i + 2):
        #                             if 0 <= k < len(differences):
        #                                 if not to_be_removed[k]:
        #                                     temp_sum_i += abs(differences[i]-differences[k])
        #                                     temp_count_i += 1
        #                         for k in range(j - 2, j + 2):
        #                             if 0 <= k < len(differences):
        #                                 if not to_be_removed[k]:
        #                                     temp_sum_j += abs(differences[j]-differences[k])
        #                                     temp_count_j += 1
        #                         if temp_sum_i / temp_count_i > temp_sum_j / temp_count_j:
        #                             to_be_removed[i] = True
        #                             to_be_removed_indices.append(i)
        #                         else:
        #                             to_be_removed[j] = True
        #                             to_be_removed_indices.append(j)
        # for i in range(len(differences)-1, -1, -1):
        #     if to_be_removed[i]:
        #         del differences[i]
        # return differences
        normalized_differences = []
        comparison_queue = Queue(size=4)
        count = 0
        while len(comparison_queue.data) < 4 and count + 7 < len(differences):

            moving_mean_dif = statistics.mean(differences[count: count + 7])
            if 0.75 * moving_mean_dif <= differences[count] <= 1.25 * moving_mean_dif:
                comparison_queue.insert_element(differences[count])
            count += 1

        for i in range(0, len(differences)):
            if len(comparison_queue.data) > 3:
                moving_mean = statistics.mean(comparison_queue.data)
                if 0.85 * moving_mean <= differences[i] <= 1.15 * moving_mean:
                    normalized_differences.append(differences[i])
                    comparison_queue.insert_element(differences[i])
                elif 0.75 * moving_mean <= differences[i] <= 1.25 * moving_mean:
                    comparison_queue.insert_element(differences[i])
        return normalized_differences

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
        """Returns the standard deviation of successive differences of beats in milliseconds"""
        diff = []
        for i in range(0, len(normalized_differences) - 1):
            diff.append(abs(normalized_differences[i] - normalized_differences[i + 1]))
        diff = diff / self.fs * 1000
        return statistics.stdev(diff)

    def __rmssd(self, normalized_differences):
        """Returns RMSSD metric, by using nn intervals (computed in milliseconds)."""
        temp_sum = 0
        normalized_differences_sec = []
        for diff in normalized_differences:
            normalized_differences_sec.append(diff / self.fs)
        for i in range(0, len(normalized_differences_sec) - 1):
            temp_sum += (normalized_differences_sec[i + 1] - normalized_differences_sec[i]) ** 2
        sz = len(normalized_differences_sec)
        if sz > 1:
            return math.sqrt(temp_sum / (sz - 1)) * 1000
        else:
            return 0

    def __hrmaxmin(self, normalized_differences):
        """Returns the difference between the max and min heartbeat, in beats per minute."""
        min_diff = 100000
        max_diff = 0
        for i in range(0, len(normalized_differences)):
            if normalized_differences[i] > max_diff:
                max_diff = normalized_differences[i]
            if normalized_differences[i] < min_diff:
                min_diff = normalized_differences[i]
        max_temp = 60/(max_diff/self.fs)
        min_temp = 60/(min_diff/self.fs)
        return min_temp - max_temp

    def __extract_ultra_short(self, data_series) -> list:
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
        def count_elements(seq, st) -> dict:
            """Tally elements from `seq`."""
            histo = {}
            for i in seq:
                ind = int(i / st)
                histo[ind] = histo.get(int(ind), 0) + 1
            return histo

        if self.fs <= 125:
            step = 1
        else:
            step = int(8 / (1000 / self.fs))
        hist = count_elements(seq=normalized_differences, st=step)
        hist_list = list(hist)
        temp_max = 0
        for val in hist_list:
            if hist[val] > temp_max:
                temp_max = hist[val]
        return temp_max / len(normalized_differences)

    def __poincare(self, normalized_differences):
        """Returns poincare metrics, as tuple (sd1, sd2, ratio, area)."""
        # difs = normalized_differences / self.fs
        # res = pyhrv.nonlinear.poincare(nni=difs, show=False)
        # return res["sd1"], res["sd2"], res["sd_ratio"], res["ellipse_area"]
        # # sd1 is equal to (0.5 * rmssd) ^ 0.5
        nn = normalized_differences / self.fs * 1000
        x1 = np.asarray(nn[:-1])
        x2 = np.asarray(nn[1:])
        sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
        area = np.pi * sd1 * sd2
        if sd1 > 0:
            ratio = sd2 / sd1
        else:
            ratio = 0
        return sd1, sd2, ratio, area

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

        window_samples = int(window * self.fs / self.factor)
        approximate_energy = []
        for i in range(0, len(data_array) - int(window_samples / 2), int(window_samples / 2)):
            if i+window_samples < len(data_array):
                approximate_energy.append(approximate_entropy(data_arr=data_array[i:i+window_samples], m2=m, r1=r))
            else:
                # approximate_energy.append(approximate_entropy(data_arr=data_array[i:], m2=m, r1=r))
                break
        mn = statistics.mean(approximate_energy)
        sd = statistics.stdev(approximate_energy)
        return mn, sd

    def __sample_entropy_mean_and_std(self, data_array, r, window: int):
        """Returns the mean and standard deviation of sample energy, in a tuple."""

        def sampen(data, m, r1):
            n = len(data)
            # b = 0.0
            # a = 0.0

            # Split time series and save all templates of length m
            xmi = np.array([data[ii: ii + m] for ii in range(n - m)])
            xmj = np.array([data[ii: ii + m] for ii in range(n - m + 1)])

            # Save all matches minus the self-match, compute B
            b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r1) - 1 for xmii in xmi])

            # Similar for computing A
            m += 1
            xm = np.array([data[ii: ii + m] for ii in range(n - m + 1)])

            a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r1) - 1 for xmi in xm])

            # Return SampEn
            del xmi, xmj, xm
            gc.collect()
            return -np.log(a / b)

        window_samples = int(window * self.fs / self.factor)
        sample_entropy = []
        for i in range(0, len(data_array) - int(window_samples / 2), int(window_samples / 2)):
            if i + window_samples < len(data_array):
                sample_entropy.append(sampen(data=data_array[i:i + window_samples], r1=r, m=2))
            else:
                # sample_entropy.append(sampen(data=data_array[i:], r1=r, m=2))
                break
        mn = statistics.mean(sample_entropy)
        sd = statistics.stdev(sample_entropy)
        return mn, sd

    def __approximate_entropy(self, data_arr, m2, r1):
        data_arr = np.array(data_arr)
        n = data_arr.shape[0]

        def _phi(m1):
            z = n - m1 + 1.0
            x = np.array([data_arr[ii:ii + m1] for ii in range(int(z))])
            x2 = np.repeat(x[:, np.newaxis], 1, axis=2)
            c = np.sum(np.absolute(x - x2).max(axis=2) <= r1, axis=0) / z
            return np.log(c).sum() / z

        return abs(_phi(m2 + 1) - _phi(m2))

    def __sampen(self, data, m, r1):
        n = len(data)
        # b = 0.0
        # a = 0.0

        # Split time series and save all templates of length m
        xmi = np.array([data[ii: ii + m] for ii in range(n - m)])
        xmj = np.array([data[ii: ii + m] for ii in range(n - m + 1)])

        # Save all matches minus the self-match, compute B
        b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r1) - 1 for xmii in xmi])

        # Similar for computing A
        m += 1
        xm = np.array([data[ii: ii + m] for ii in range(n - m + 1)])

        a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r1) - 1 for xmi in xm])

        # Return SampEn
        return -np.log(a / b)

    def __normalize_and_downscale(self, data_series, factor):
        """Normalizes a data series list, so that max mag is 1 and also downscales it by a factor.
        Returns it as a list."""
        output = []
        counter = factor
        for data in data_series:
            if counter % factor == 0:
                output.append(data)
            counter += 1
        return output

