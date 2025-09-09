import shutil

import pandas as pd

df = pd.read_parquet("hf://datasets/allenai/ZebraLogicBench-private/grid_mode/test-00000-of-00001.parquet")

# store the dataset in the local directory
df.to_parquet("grid_mode_test.parquet")
# copy the dataset to the data directory
# shutil.copy("grid_mode_test.parquet", "../data/grid_mode_test.parquet")


