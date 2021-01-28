# Open .dat files downloaded from the internet

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class SimpleExample:
    data_frame = pd.read_csv("16265.csv", header=None, names=['time_step', 'first_measurement', 'second_measurement'],
                             delim_whitespace=True)
    col_b = data_frame["first_measurement"]
    sampling_frequency = (data_frame.time_step.size - 1)/data_frame.time_step.values[-1]
    print(f"Sampling_frequency is {sampling_frequency} Hz")
    mini_col_b = col_b[:3840]
    print(type(mini_col_b))
    # mini_col_b.to_csv("./example.csv")
    # print(type(mini_col_b))
    mini_col_b.plot()
    plt.show()
