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
if len(args) > 1:
    file = args[1]
    txt = file.split(sep='/')[2]
    # print(f"File number {count} out of {len(files)}: {txt}")
    myExtraction = Extraction(file)
    # print(file)
    # print(txt)
    res = pd.DataFrame(myExtraction.extract_short_features())
    # count += 1
    output = "./extracted_features/final_extracted_"+txt[:-4]+"_freq_add_on.csv"
    res.to_csv(path_or_buf=output)
    del res
    del myExtraction
    gc.collect()
# res = pd.concat(dataframes, ignore_index=True)


