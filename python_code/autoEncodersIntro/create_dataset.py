import glob
from separate_measurements import Separation
from r_peaks_detection import PeaksDetection
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ULTRA_SHORT_TIME_WINDOW = 30


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


class DataCreation:

    def __init__(self, dir1):
        self.dir1 = dir1

    def creation(self):
        # print(glob.glob(self.dir1 + "/*.csv"))
        sample_files = glob.glob(self.dir1 + "/*.csv")
        final_samples_list = []
        for file in sample_files:
            print(file)
            (samples, fs) = Separation(time_duration=ULTRA_SHORT_TIME_WINDOW, file=file) \
                .separate_ecgs()
            len1 = len(samples)
            # We use only 10% of the original data
            # samples = samples[:int(len1 / 10)]
            # samples = samples[0:10]
            for sample in samples:
                r_peaks = PeaksDetection(fs=fs, data_series=sample).detect_peaks()
                differences = []
                for i in range(1, len(r_peaks)):
                    differences.append(r_peaks[i] - r_peaks[i - 1])
                normalized_differences = self.__normalized_differences(differences=differences)
                # if len(normalized_differences) == len(differences):
                my_sample_np = sample.to_numpy()
                for i in range(0, len(r_peaks) - 1):
                    my_sample = my_sample_np[r_peaks[i]:r_peaks[i+1]]
                    my_len = len(my_sample)
                    if my_len > 10:
                        x_interp = np.arange(140)
                        xp_interp = np.arange(start=0, stop=139.999999999, step=140/my_len)
                        if len(xp_interp) == len(my_sample):
                            # Normalize between 0 and 1

                            interpolated = np.interp(x=x_interp, xp=xp_interp, fp=my_sample)
                            min_value = interpolated.min()
                            max_value = interpolated.max()
                            interpolated = (interpolated - min_value) / (max_value - min_value)
                            # Uncomment below to plot

                            # plt.grid()
                            # plt.plot(x_interp, interpolated)
                            # plt.title("A Normal ECG")
                            # plt.show()

                            final_samples_list.append(interpolated.tolist())
                        else:
                            print("Error here with the lengths of samples.")
        df = pd.DataFrame(final_samples_list)
        output = "./testCadDataset.csv"
        df.to_csv(output)

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
