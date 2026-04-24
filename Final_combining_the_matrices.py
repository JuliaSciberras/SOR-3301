# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 11:35:00 2025

@author: Julia Sciberras
"""

import pandas as pd
from pathlib import Path

#  Set your input and output files

input_files = {
    "0730": "weekday_distance_time_at730am.xlsx",
    "1130": "weekday_distance_time_at1130am.xlsx",
    "1330": "weekday_distance_time_at1330.xlsx"
}

output_file = "enter_output_file.xlsx"

# clean row/column labels to 0..149 

def set_0_149_labels(df, n=150):
    if df.shape != (n, n):
        print(f"WARNING: expected shape ({n}, {n}), got {df.shape}")
    df.index = range(n)
    df.columns = range(n)
    return df

#  Process all files and write to one Excel workbook

with pd.ExcelWriter(output_file) as writer:   
    for tag, file_path in input_files.items():
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"WARNING: file not found: {file_path}")
            continue

        # Read Excel structure
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        # Skip the last 2 sheets
        use_sheets = sheet_names[:-2]

        print(f"Processing {file_path.name}: using sheets {use_sheets}")

        for sheet in use_sheets:
            df = pd.read_excel(xls, sheet_name=sheet, index_col=0)

            # Ensure labels are 0..149
            df = set_0_149_labels(df)

            # Excel sheet names must be max 31 chars
            out_sheet_name = f"{tag}_{sheet}"[:31]

            df.to_excel(writer, sheet_name=out_sheet_name)

print(f"Done! Combined file saved as: {output_file}")
