import pandas as pd
import numpy as np

# === SETTINGS ===
input_file = "all_travel_times_combined.xlsx"
output_file = "cell_percentiles_output.xlsx"

# Load all sheets with headers
sheets = pd.read_excel(input_file, sheet_name=None, header=0, index_col=0)

# Exclude the last 2 sheets
sheet_names = list(sheets.keys())[:-2]

# We will collect values for each cell across sheets
all_data = []

for sheet_name in sheet_names:
    df = sheets[sheet_name]
    block = df.iloc[:11, :11]  
    all_data.append(block)

# Stack all sheets into a 3D array: (num_sheets, 6, 6)
data_3d = np.stack([df.to_numpy() for df in all_data], axis=0)

# Compute percentiles PER CELL across the sheet dimension (axis=0)
p25 = np.percentile(data_3d, 25, axis=0)
p50 = np.percentile(data_3d, 50, axis=0)
p75 = np.percentile(data_3d, 75, axis=0)

# Convert arrays back to DataFrames with correct labels
index_labels = all_data[0].index
col_labels = all_data[0].columns

p25_df = pd.DataFrame(p25, index=index_labels, columns=col_labels)
p50_df = pd.DataFrame(p50, index=index_labels, columns=col_labels)
p75_df = pd.DataFrame(p75, index=index_labels, columns=col_labels)

# Save to Excel
with pd.ExcelWriter(output_file) as writer:
    p25_df.to_excel(writer, sheet_name="P25")
    p50_df.to_excel(writer, sheet_name="P50")
    p75_df.to_excel(writer, sheet_name="P75")

print("Done. Per-cell percentiles written to:", output_file)

