import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


"""Usage
Concatenate CSV files
"""


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='.')
parser.add_argument('--dirs', nargs='+',
                    help='Specify one or more directories')
opt = parser.parse_args()

# ['Tray1', 'Tray2']
# print(opt.dirs)

main_folder = Path(opt.path)
meta_filepath = main_folder / 'metadata.csv'

for i, subfolder in enumerate(opt.dirs):
    submain_folder = main_folder / subfolder
    submain_meta_filepath = submain_folder / 'metadata.csv'
    submeta_df = pd.read_csv(submain_meta_filepath)
    if i == 0:
        meta_df = submeta_df
    else:
        meta_df = meta_df.append(submeta_df, ignore_index=True)

meta_df.to_csv('metadata.csv')
