import sys
from extract_features import Extraction
import glob
import pandas as pd
import gc

file_titles = "./extracted_features/final/*.csv"
files = glob.glob(file_titles)
file_titles_no_cad = "./extracted_features/final/final_extracted_1*.csv"
files_no_cad = glob.glob(file_titles_no_cad)
file_titles_cad = "./extracted_features/final/final_extracted_s*.csv"
files_cad = glob.glob(file_titles_cad)
print(f"Total number of files: {len(files)}")
print(f"Total number of CAD files: {len(files_cad)}")
print(f"Total number of NO CAD files: {len(files_no_cad)}")

dataframes_cad = []
for file in files_cad:
    dataframes_cad.append(pd.read_csv(file))
res_cad = pd.concat(dataframes_cad, ignore_index=True)
output = "./extracted_features/final/" + "final_cad.csv"
res_cad.to_csv(path_or_buf=output)

dataframes_no_cad = []
for file in files_no_cad:
    dataframes_no_cad.append(pd.read_csv(file))
res_no_cad = pd.concat(dataframes_no_cad, ignore_index=True)
output = "./extracted_features/final/" + "final_no_cad.csv"
res_no_cad.to_csv(path_or_buf=output)

dataframes = []
for file in files:
    dataframes.append(pd.read_csv(file))
res = pd.concat(dataframes, ignore_index=True)
output = "./extracted_features/final/" + "final_full.csv"
res.to_csv(path_or_buf=output)
