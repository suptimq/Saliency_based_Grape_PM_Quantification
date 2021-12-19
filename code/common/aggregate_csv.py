import os
import glob
import argparse
import xlsxwriter
import pandas as pd
from pathlib import Path


"""Usage
Transfer CSV files to EXCEL sheet tabs and aggregate them into a EXCEL file
"""


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
opt = parser.parse_args()
print(opt)

main_folder = Path(opt.path)
excel_filepath = main_folder / 'multi_sheet.xlsx'
csv_filenames = glob.glob(str(main_folder / '*.csv'))

excel_writer = pd.ExcelWriter(excel_filepath, engine='xlsxwriter')

for csv_filename in csv_filenames:
    csv_filepath = main_folder / csv_filename
    curr_df = pd.read_csv(csv_filepath, index_col=False)
    curr_df.to_excel(
        excel_writer, sheet_name=os.path.basename(csv_filename)[:-4])

excel_writer.save()
print(f'Saved {excel_filepath}')
