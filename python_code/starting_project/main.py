import sys
from extract_features import Extraction
import glob
import pandas as pd
import gc

# file_titles = "./samples/*.csv"
# files = glob.glob(file_titles)
# print(f"Total number of files: {len(files)}")
# count = 1
args = sys.argv
# if len(args) > 1:
print("Hello World")
if 1 == 1:
    # file = args[1]
    file = "./cad_samples/fifth/s20551.csv"
    txt = file.split(sep='/')[3]
    print(txt)
    # print(f"File number {count} out of {len(files)}: {txt}")
    # print(file)
    # print(txt)
    res = pd.DataFrame(Extraction(file).extract_windowed_short_features(print_message=False))
    # count += 1
    output = "./extracted_features/short_updated/short_updated_extracted_"+txt
    res.to_csv(path_or_buf=output)
    del res
    gc.collect()
# res = pd.concat(dataframes, ignore_index=True)


