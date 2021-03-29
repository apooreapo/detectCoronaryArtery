
# Created on: 11/18/2019
# Purpose: Re-create Pan-Tompkins algorithm for R-peak detections
# Method is based on the original Pan-Tompkins method and Hooman Sedghamiz's
# Reference: https://www.researchgate.net/publication/313673153_Matlab_Implementation_of_Pan_Tompkins_ECG_QRS_detector
# Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from scipy import signal
import math


class PanTompkins:

    def __init__(self, fs: float, data_series: pd.core.series.Series, print_message=False):
        self.fs = fs
        self.data_series = data_series
        self.print_message = print_message

    def __interpolate(self, input_array, step: float):
        """Same as matlab's interpl function"""
        input_index = []
        count = 0
        input_array_size = len(input_array)
        for ii in np.arange(0.0, float(input_array_size) - 1.0 + 0.00001, step):
            input_index.append(ii)
            # print(ii)
        output = []
        for ind in input_index:
            count += 1
            fl = int(math.floor(ind))
            ce = int(math.ceil(ind))
            down = ind - math.floor(ind)
            if ind == float(input_array_size - 1):
                output.append(input_array[-1])
            else:
                output.append(float(down)*(input_array[ce] - input_array[fl]) + input_array[fl])
        return output

    def pan_tompkins(self):

        # Import Data
        #     df = pd.read_csv("./myECG.csv", header=None, sep=",",
        #                                      names=["measurement", "useless"])
        ecg = self.data_series.to_numpy()
        fs = self.fs
        # print(df)
        # df = pd.read_excel(r"/Users/lokyiuto/Dropbox/Learning Python/pan_tompkins/data.xlsx", header=None)

        # ecg = df.measurement.tolist()
        # ecg = np.asarray(ecg)

        time = np.arange(0, len(ecg) / fs, 1 / fs)

        # plt.plot(time, ecg, label='unfiltered')
        # plt.show()

        # Remove mean
        mn = np.mean(ecg)
        ecg = ecg - mn

        # Bandpass Filter
        fc = np.array([5, 15])
        fc = fc * 2 / fs

        N = 3
        b, a = signal.butter(N, fc, 'bandpass')
        ecg_f = signal.filtfilt(b, a, ecg)
        ecg_f = ecg_f / max(ecg_f)
        #
        # plt.plot(time, ecg_f)
        # plt.show()

        # Derivative Filter
        factor1 = fs / 8.0
        derArray1 = [1.0*factor1, 2.0*factor1, 0, -2*factor1, -1*factor1]
        b1 = self.__interpolate(derArray1, float(4.0 / (fs / 40.0)))
        ecg_d = signal.filtfilt(b1, 1, ecg_f)
        ecg_d = ecg_d / max(ecg_d)
        # plt.plot(time, ecg_d)
        # plt.show()

        # Squaring nonlinearly to enhance dominant peaks
        ecg_s = np.square(ecg_d)
        # plt.plot(time, ecg_s)
        # plt.show()

        # Moving Average
        array_c = np.divide(np.ones(int(np.round(0.15 * fs))), np.round(0.15 * fs))

        ecg_m = np.convolve(ecg_s, array_c)
        time2 = np.arange(0, len(ecg_m) / fs, 1 / fs)

        # plt.plot(time2, ecg_m)
        # plt.show()

        # Fiducial Marks
        locs_R, _ = signal.find_peaks(ecg_m, distance=np.round(0.2 * fs))
        pks_R = ecg_m[locs_R]

        # plt.plot(ecg_m)
        # plt.plot(locs_R, ecg_m[locs_R], "x")
        # plt.show()


        # Threshold Adjustment
        thresh_sig_f = max(ecg_f[:2 * int(fs) - 1]) * 1.0 / 3.0
        thresh_noi_f = st.mean(ecg_f[:2 * int(fs) - 1]) / 2.0
        sig_lev_f = thresh_sig_f
        noi_lev_f = thresh_noi_f

        thresh_sig_m = max(ecg_m[:2 * int(fs) - 1]) * 1.0 / 3.0
        thresh_noi_m = st.mean(ecg_m[:2 * int(fs) - 1]) / 2.0
        sig_lev_m = thresh_sig_m
        noi_lev_m = thresh_sig_m

        beat_C = 0
        beat_C1 = 0
        selected_meanRR = 0
        meanRR = 0
        skip = 0
        ser_back = 0
        pkNum = len(locs_R)
        qrs_i = np.zeros(pkNum)
        qrs_i_raw = np.zeros(pkNum)

        for i in range(len(locs_R)):
            y_i = 0
            x_i = 0
            # locate the corresponding peak in the filter signal
            if locs_R[i] - np.round(0.15 * fs) >= 1 and locs_R[i] <= len(ecg_f):
                x_i = np.argmax(ecg_f[locs_R[i] - int(np.round(0.15 * fs)): locs_R[i] + 1])
                y_i = max(ecg_f[locs_R[i] - int(np.round(0.15 * fs)): locs_R[i] + 1])

            else:
                if i == 0:
                    x_i = np.argmax(ecg_f[:locs_R[i] + 1])
                    y_i = max(ecg_f[:locs_R[i] + 1])
                    ser_back = 1
                elif locs_R[i] >= len(ecg_f) - 1:
                    x_i = np.argmax(ecg_f[locs_R[i] - int(np.round(0.15 * fs)):])
                    y_i = max(ecg_f[locs_R[i] - int(np.round(0.15 * fs)):])

            # update the heart rate
            if beat_C >= 9:
                diffRR = np.diff(qrs_i[beat_C - 9:beat_C - 1])
                meanRR = st.mean(diffRR)
                lastRR = diffRR[-1]

                if lastRR <= 0.92 * meanRR or lastRR >= 1.16 * meanRR:
                    thresh_sig_m *= 0.5
                    thresh_sig_f *= 0.5

                else:
                    selected_meanRR = meanRR

            # Search back for missed QRS complexes
            if selected_meanRR > 0:
                test_m = selected_meanRR
            elif meanRR > 0 and selected_meanRR == 0:
                test_m = meanRR
            else:
                test_m = 0

            if test_m > 0:
                if locs_R[i] - qrs_i[beat_C - 1] >= np.round(1.66 * test_m):  # QRS is missing
                    pks_temp = max(ecg_m[int(qrs_i[beat_C - 1]) + int(np.round(0.2 * fs)):
                                         int(locs_R[i]) - int(np.round(0.2 * fs)) + 1])
                    locs_temp = np.argmax(ecg_m[int(qrs_i[beat_C - 1]) + int(np.round(0.2 * fs)): int(locs_R[i]) - int(np.round(0.2 * fs)) + 1])
                    locs_temp = int(qrs_i[beat_C - 1]) + int(np.round(0.2 * fs)) + int(locs_temp) - 1

                    if pks_temp > thresh_noi_m:
                        beat_C += 1
                        qrs_i[beat_C - 1] = locs_temp

                    if locs_temp <= len(ecg_f) - 1:
                        x_i_temp = np.argmax(ecg_f[locs_temp - int(np.round(0.15 * fs)): locs_temp])
                        y_i_temp = max(ecg_f[locs_temp - int(np.round(0.15 * fs)): locs_temp])

                    else:
                        x_i_temp = np.argmax(ecg_f[locs_temp - int(np.round(0.15 * fs)):])
                        y_i_temp = max(ecg_f[locs_temp - int(np.round(0.15 * fs)):])

                    if y_i_temp > thresh_noi_f:  # band-pass signal threshold
                        beat_C1 += 1
                        qrs_i_raw[beat_C - 1] = locs_temp - int(np.round(0.15 * fs)) + x_i_temp
                        sig_lev_f = 0.25 * y_i_temp + 0.75 * sig_lev_f

                    sig_lev_m = 0.25 * pks_temp + 0.75 * sig_lev_m

            # find noise and QRS peak
            if pks_R[i] >= thresh_sig_m:
                #  if no qrs within 360ms of the previous QRS, see if T waves exist
                if beat_C >= 3:
                    if locs_R[i] - int(qrs_i[beat_C]) <= int(np.round(0.36 * fs)):
                        slope1 = st.mean(
                            np.diff(ecg_m[locs_R[i] - int(np.round(0.36 * fs)): locs_R[i] + 1]))  # mean slope at the location
                        slope2 = st.mean(np.diff(ecg_m[int(qrs_i[beat_C]) - int(np.round(0.36 * fs)):int(qrs_i[beat_C]) + 1]))
                        if abs(slope1) <= abs(0.5 * slope2):
                            skip = 1

                            # adjust noise level
                            noi_lev_f = 0.125 * y_i + 0.875 * noi_lev_f
                            noi_lev_m = 0.125 * pks_R[i] + 0.875 * noi_lev_m
                        else:
                            skip = 0

                if skip == 0:
                    beat_C += 1
                    qrs_i[beat_C - 1] = locs_R[i]

                    if y_i >= thresh_sig_f:
                        beat_C1 += 1
                        if ser_back == 1:
                            qrs_i_raw[beat_C1 - 1] = x_i
                        else:
                            qrs_i_raw[beat_C1 - 1] = locs_R[i] - int(np.round(0.15 * fs)) + x_i

                        sig_lev_f = 0.125 * y_i + 0.875 * sig_lev_f

                    sig_lev_m = 0.125 * y_i + 0.875 * sig_lev_m

            elif thresh_noi_m <= pks_R[i] and pks_R[i] < thresh_sig_m:
                noi_lev_f = 0.125 * y_i + 0.875 * noi_lev_f
                noi_lev_m = 0.125 * pks_R[i] + 0.875 * noi_lev_m

            elif pks_R[i] <= thresh_noi_m:
                noi_lev_f = 0.125 * y_i + 0.875 * noi_lev_f
                noi_lev_m = 0.125 * pks_R[i] + 0.875 * noi_lev_m

            # adjust the threshold with SNR
            if noi_lev_m != 0 or sig_lev_m != 0:
                thresh_sig_m = noi_lev_m + 0.25 * abs(sig_lev_m - noi_lev_m)
                thresh_noi_m = 0.5 * thresh_sig_m

            if noi_lev_f != 0 or sig_lev_f != 0:
                thresh_sig_f = noi_lev_f + 0.25 * abs(sig_lev_f - noi_lev_f)
                thresh_noi_f = 0.5 * thresh_sig_f

            # Reset parameter
            skip = 0
            ser_back = 0

        qrs_i_raw = np.trim_zeros(qrs_i_raw)

        qrs_i_raw_int = np.ndarray(len(qrs_i_raw), dtype=int)
        # print(qrs_i_raw_int)
        count = -1
        for qrs in qrs_i_raw:
            count += 1
            qrs_i_raw_int[count] = int(qrs)

        res = qrs_i_raw_int.tolist()
        # print(res)

        if self.print_message:
            plt.plot(ecg_f)
            plt.plot(qrs_i_raw_int, ecg_f[qrs_i_raw_int], "x")
            plt.show()

        return res



