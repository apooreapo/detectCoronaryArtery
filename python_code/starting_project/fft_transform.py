import numpy as np
import matplotlib.pyplot as plotter


class FFTTransform:

    def __init__(self, fs, data_series):
        self.fs = fs
        self.data_series = data_series
        max1 = max(data_series)
        self.sampling_interval = 1 / self.fs
        begin_time = 0
        end_time = (len(self.data_series)) * self.sampling_interval
        self.time = np.arange(begin_time, end_time, self.sampling_interval)

        self.np_arr = np.asarray(self.data_series, dtype=np.float)
        self.np_arr /= max1 # normalize the data so that they have max value equal to 1
        self.fourier_transform = np.fft.fft(self.np_arr) / len(self.data_series)
        self.fourier_transform = self.fourier_transform[range(int(len(self.np_arr) / 2))]  # fft
        tp_count = len(self.np_arr)  # The length of the sample
        values = np.arange(int(tp_count / 2))  # We only need half frequencies, the other half are redundant
        self.time_period = tp_count / self.fs  # The full time of sample in seconds.
        self.resolution = 1 / self.time_period  # Freq resolution: 1/ time_period
        self.frequencies = values / self.time_period   # The array of different frequencies, in Hz.
        # print(self.frequencies)


    def plot(self):
        """Plots fft of a sample."""
        print(self.frequencies)
        print(abs(self.fourier_transform))

        figure, axis = plotter.subplots(2, 1)
        plotter.subplots_adjust(hspace=1)
        axis[0].set_title('ECG')
        axis[0].plot(self.time, self.np_arr)
        axis[0].set_xlabel('Time')
        axis[0].set_ylabel('Amplitude')

        axis[1].set_title('Fourier transform depicting the frequency components')
        axis[1].plot(self.frequencies, abs(self.fourier_transform))
        axis[1].set_xlabel('Frequency')
        axis[1].set_ylabel('Amplitude')
        plotter.show()

    def vlf_band(self) -> tuple:
        """Calculates the energy of the Very Low Frequency Band (0.0033 - 0.04 Hz)
        Returns the vlf energy, its percentage to the full energy and the peak energy of the band
        in a tuple: 'energy, percentage, peak'."""
        lower_limit = 0.0033 * self.time_period
        upper_limit = 0.04 * self.time_period
        peak_energy = 0.0
        peak_energy_position = 0
        temp_sum = 0
        full_sum = 0
        for i in range(0, int(len(self.data_series) / 2)):
            full_sum += abs(self.fourier_transform[i]) ** 2
        for i in range(0, len(self.np_arr)):
            if i >= lower_limit:
                if i <= upper_limit:
                    temp1 = abs(self.fourier_transform[i]) ** 2
                    temp_sum += temp1
                    if peak_energy < temp1:
                        peak_energy = temp1
                        peak_energy_position = i / self.time_period
                else:
                    break
        percentage = temp_sum / full_sum
        # print(f"VLF band energy is {round(percentage * 100, 4)}% of the full energy.")
        # print(f"VLF energy is {temp_sum}.")
        temp_sum *= self.resolution  # Resolution is equivalent to dx in our computation sum
        return temp_sum, percentage, peak_energy_position

    def lf_band(self) -> tuple:
        """Calculates the energy of the Low Frequency Band (0.04 - 0.15 Hz)
        Returns the lf energy, its percentage to the full energy and the peak energy of the band
        in a tuple: 'energy, percentage, peak'."""
        lower_limit = 0.04 * self.time_period
        upper_limit = 0.15 * self.time_period
        temp_sum = 0
        full_sum = 0
        peak_energy = 0.0
        peak_energy_position = 0
        for i in range(0, int(len(self.data_series) / 2)):
            full_sum += abs(self.fourier_transform[i]) ** 2
        for i in range(0, len(self.np_arr)):
            if i > lower_limit:
                if i <= upper_limit:
                    temp1 = abs(self.fourier_transform[i]) ** 2
                    temp_sum += temp1
                    if peak_energy < temp1:
                        peak_energy = temp1
                        peak_energy_position = i / self.time_period
                else:
                    break
        percentage = temp_sum / full_sum
        # print(f"LF band energy is {round(percentage * 100, 2)}% of the full energy.")
        temp_sum *= self.resolution  # Resolution is equivalent to dx in our computation sum
        return temp_sum, percentage, peak_energy_position

    def hf_band(self) -> tuple:
        """Calculates the energy of the High Frequency Band (0.15 - 0.40 Hz)
        Returns the lf energy, its percentage to the full energy and the peak energy of the band
        in a tuple: 'energy, percentage, peak'."""
        lower_limit = 0.15 * self.time_period
        upper_limit = 0.401 * self.time_period
        temp_sum = 0
        full_sum = 0
        peak_energy = 0.0
        peak_energy_position = 0
        for i in range(0, int(len(self.data_series) / 2)):
            full_sum += abs(self.fourier_transform[i]) ** 2
        for i in range(0, len(self.np_arr)):
            if i > lower_limit:
                if i <= upper_limit:
                    temp1 = abs(self.fourier_transform[i]) ** 2
                    temp_sum += temp1
                    if peak_energy < temp1:
                        peak_energy = temp1
                        peak_energy_position = i / self.time_period
                else:
                    break
        percentage = temp_sum / full_sum
        # print(f"HF band energy is {round(percentage * 100, 2)}% of the full energy.")
        temp_sum *= self.resolution  # Resolution is equivalent to dx in our computation sum
        return temp_sum, percentage, peak_energy_position
    #
    # def __downscale(self, data_series, factor):
    #     output = []
    #     for i in range(0, len(data_series), factor):
    #         output.append(data_series[i])
    #     return output
    #
    # def normalize_data_series(self, data_series, factor):
    #     """Normalizes a data series list, so that max mag is 1 and also downscales it by a factor.
    #     Returns it as a numpy array."""
    #     max1 = max(data_series)
    #     output = []
    #     for i in range(0, len(data_series), factor):
    #         output.append(data_series[i] / max1)
    #     np_arr = np.asarray(output, dtype=np.float)
    #     return np_arr


class FFTWindowedTransform:
    def __init__(self, fs, data_series):
        self.fs = fs
        self.data_series = data_series
        max1 = []
        for ultra_short_data in data_series:
            max1.append(max(ultra_short_data))
        self.sampling_interval = 1 / self.fs
        begin_time = 0
        end_time = (len(self.data_series[0])) * self.sampling_interval
        self.time = np.arange(begin_time, end_time, self.sampling_interval)

        self.np_arr = []
        for ii in range(len(data_series)):
            self.np_arr.append(np.asarray(data_series[ii], dtype=np.float))
            self.np_arr[ii] /= max1[ii]  # normalize the data so that they have max value equal to 1
        self.fourier_transform = []
        tp_count = []
        values = []
        self.time_period = []
        self.resolution = []
        self.frequencies = []
        for ii in range(len(data_series)):
            self.fourier_transform.append(np.fft.fft(self.np_arr[ii]) / len(self.data_series[ii]))
            self.fourier_transform[ii] = self.fourier_transform[ii][range(int(len(self.np_arr[ii]) / 2))]  # fft
            tp_count.append(len(self.np_arr[ii]))  # The length of the sample
            values.append(np.arange(int(tp_count[ii] / 2)))  # We only need half frequencies

            self.time_period.append(tp_count[ii] / self.fs)  # The full time of sample in seconds.
            self.resolution.append(1 / self.time_period[ii])  # Freq resolution: 1/ time_period
            self.frequencies.append(values[ii] / self.time_period[ii])  # The array of different frequencies, in Hz.

    def custom_band(self, lower_limit, upper_limit) -> tuple:
        """Calculates the energy of the Low Frequency Band (lower_limit - upper_limit Hz)
        Returns the energy, its percentage to the full energy and the peak energy of the band
        in a tuple: 'energy, percentage, peak'."""
        lower_limit *= self.time_period[0]
        upper_limit *= self.time_period[0]
        my_dict = {}
        temp_sum = 0
        full_sum = 0
        for i in range(0, len(self.data_series)):
            for j in range(0, int(len(self.data_series[i]) / 2)):
                full_sum += abs(self.fourier_transform[i][j]) ** 2

        # Initialize the dictionary with the fourier sums
        for i in range(0, len(self.np_arr[0])):
            if i > lower_limit:
                if i <= upper_limit:
                    my_dict[i] = 0
                else:
                    break

        # Start adding the sums
        for i in range(0, len(self.np_arr)):
            for j in range(0, len(self.np_arr[i])):
                if j > lower_limit:
                    if j <= upper_limit:
                        temp1 = abs(self.fourier_transform[i][j]) ** 2
                        temp_sum += temp1
                        my_dict[j] += temp1
                    else:
                        break
        percentage = temp_sum / full_sum
        max_energy = 0.0
        max_energy_loc = 0
        for ind in my_dict:
            temp2 = my_dict[ind]
            if temp2 > max_energy:
                max_energy = temp2
                max_energy_loc = float(ind) / self.time_period[0]
        # print(f"LF band energy is {round(percentage * 100, 2)}% of the full energy.")
        temp_sum *= self.resolution[0]  # Resolution is equivalent to dx in our computation sum
        return temp_sum, percentage, max_energy_loc
