from extract_features import Extraction
import glob
import pandas as pd

files = glob.glob("./samples/*.csv")
dataframes = []
for file in files:
    dataframes.append(pd.DataFrame(Extraction(file).extract_short_features()))
res = pd.concat(dataframes, ignore_index=True)

res.to_csv(path_or_buf="./example_features_output.csv")
# print(pd.DataFrame(dicts[0]))
# print(pd.DataFrame(dicts[1]))