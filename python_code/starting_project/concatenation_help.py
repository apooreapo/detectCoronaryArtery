import pandas as pd
import glob

dir = "./extracted_features/final/"
file_titles = "./extracted_features/final/final_extracted_*freq_add_on.csv"
files = glob.glob(file_titles)
for freq_file in files:
    # print(file[0:-16])
    full_file = freq_file[0:-16] + ".csv"
    # print(full_file)
    final_name = full_file[:-4] + "_last.csv"
    # print(final_name)
    df_data = pd.read_csv(full_file)
    df_freq = pd.read_csv(freq_file)
    df_full = pd.concat([df_data, df_freq], axis=1)
    del df_full["Unnamed: 0"]
    df_full.to_csv(path_or_buf=final_name)

