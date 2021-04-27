import glob
from separate_measurements import Separation
from r_peaks_detection import PeaksDetection
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
# import tensorflow.keras
import keras
keras.models.sa

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

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

class Evaluation:

    def __init__(self, dir1):
        self.dir1 = dir1
        self.autoencoder = tensorflow.keras.models.load_model('./autoencoder')
        self.threshold = 0.04304970990934394 # Got this as (mean + 2.5 * std)

    def creation(self):
        # print(glob.glob(self.dir1 + "/*.csv"))
        sample_files = glob.glob(self.dir1 + "/*.csv")
        final_samples_list = []
        ratios = []
        for file in sample_files:
            print(file)
            (samples, fs) = Separation(time_duration=ULTRA_SHORT_TIME_WINDOW, file=file) \
                .separate_ecgs()
            len1 = len(samples)
            # We use only 10% of the original data
            samples = samples[:int(len1 / 10)]
            # samples = samples[0:10]
            for sample in samples:
                raw_data = []
                r_peaks = PeaksDetection(fs=fs, data_series=sample).detect_peaks()
                if len(r_peaks) > 1:
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
                                raw_data.append(interpolated)
                                # Uncomment below to plot

                                # plt.grid()
                                # plt.plot(x_interp, interpolated)
                                # plt.title("A Normal ECG")
                                # plt.show()


                            else:
                                print("Error here with the lengths of samples.")
                    raw_data = np.asarray(raw_data)
                    encoded_imgs = self.autoencoder.encoder(raw_data).numpy()
                    decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()
                    train_loss = tf.keras.losses.mae(decoded_imgs, raw_data)
                    # print(train_loss)
                    full_length = len(raw_data)
                    faulty_length = len(train_loss[train_loss>self.threshold])
                    # print(f"Full length:{full_length}, faulty length:{faulty_length}")
                    ratios.append(faulty_length / full_length)
                    # print("Here are the ratios")
                    # print(ratios)

                    # plt.plot(raw_data[0], 'b')
                    # plt.plot(decoded_imgs[0], 'r')
                    # plt.fill_between(np.arange(140), decoded_imgs[0], raw_data[0], color='lightcoral')
                    # plt.legend(labels=["Input", "Reconstruction", "Error"])
                    # plt.show()
        plt.hist(ratios, bins=20)
        plt.xlabel("Faulty percentage")
        plt.ylabel("No. of examples")
        plt.title("Faulty R-R intervals percentage, normal sinus ECGs")
        plt.show()

    def creation_from_csv(self, csv_path):
        fs = 512.414
        # print(glob.glob(self.dir1 + "/*.csv"))
        final_samples_list = []
        ratios = []
        samples = pd.read_csv(csv_path)
        raw_samples = samples.values
        len1 = len(raw_samples)
        # We use only 10% of the original data
        # samples = samples[:int(len1 / 10)]
        # samples = samples[0:10]
        for sample in raw_samples:
            raw_data = []
            max_val = sample.max()
            sample = sample / max_val
            # plt.grid()
            # plt.plot(np.arange(len(sample)), sample)
            # plt.title("A Normal ECG")
            # plt.show()
            r_peaks = PeaksDetection(fs=fs, data_series=pd.Series(sample)).detect_peaks()
            if len(r_peaks) > 1:
                for i in range(0, len(r_peaks) - 1):
                    my_sample = sample[r_peaks[i]:r_peaks[i+1]]
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
                            raw_data.append(interpolated)
                            # Uncomment below to plot

                            # plt.grid()
                            # plt.plot(x_interp, interpolated)
                            # plt.title("A Normal ECG")
                            # plt.show()


                        else:
                            print("Error here with the lengths of samples.")
                raw_data = np.asarray(raw_data)
                encoded_imgs = self.autoencoder.encoder(raw_data).numpy()
                decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()
                train_loss = tf.keras.losses.mae(decoded_imgs, raw_data)
                # print("Hello sample.")
                # plt.plot(np.arange(len(sample)), sample)
                # plt.show()
                # for i in range(len(raw_data)):
                #     print(train_loss[i])
                #     plt.plot(raw_data[i], 'b')
                #     plt.plot(decoded_imgs[i], 'r')
                #     plt.fill_between(np.arange(140), decoded_imgs[i], raw_data[i], color='lightcoral')
                #     plt.legend(labels=["Input", "Reconstruction", "Error"])
                #     plt.show()

                # print("Train loss array:")
                # print(train_loss)
                full_length = len(raw_data)
                faulty_length = len(train_loss[train_loss>self.threshold])
                # print(f"Full length:{full_length}, faulty length:{faulty_length}")
                ratios.append(faulty_length / full_length)
                # print(ratios[-1])
                # print("Here are the ratios")
                # print(ratios)

                # plt.plot(raw_data[0], 'b')
                # plt.plot(decoded_imgs[0], 'r')
                # plt.fill_between(np.arange(140), decoded_imgs[0], raw_data[0], color='lightcoral')
                # plt.legend(labels=["Input", "Reconstruction", "Error"])
                # plt.show()
        plt.hist(ratios, bins=20)
        plt.xlabel("Faulty percentage")
        plt.ylabel("No. of examples")
        plt.title("Faulty R-R intervals percentage, apple watch ECGs")
        plt.show()

        print(len(raw_samples))
        for i in range(len(ratios)):
            if ratios[i] < 0.5:
                print(ratios[i])
                plt.plot(np.arange(len(raw_samples[i])), raw_samples[i])
                plt.show()

