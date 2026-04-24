# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 08:43:32 2026

@author: Julia Sciberras
"""

import pandas as pd
import numpy as np



#File Paths 
matrix_file = "scenario_matrices_kmeans.xlsx"
demand_file = "customer_demands_for149customers.xlsx"



# load the data 

xls = pd.ExcelFile(matrix_file)
sheet_names = xls.sheet_names

# Assume last sheet = probability sheet
matrix_sheets = sheet_names[:-1]
probability_sheet_name = sheet_names[-1]

demands = pd.read_excel(demand_file)

#select same customers 
np.random.seed(42)

customers = list(range(1, 150))
selected_customers = np.random.choice(customers, size=49, replace=False)

selected_nodes = np.sort(np.append(0, selected_customers))

print("Selected nodes:", selected_nodes)

#process each scenario 
reduced_matrices = {}

for sheet in matrix_sheets:
    matrix = pd.read_excel(matrix_file, sheet_name=sheet, index_col=0)

    reduced_matrix = matrix.loc[selected_nodes, selected_nodes]
    reduced_matrices[sheet] = reduced_matrix

#filter demands 
filtered_demands = demands[demands['node'].isin(selected_nodes)]

#getting probability sheet 
probability_sheet = pd.read_excel(matrix_file, sheet_name=probability_sheet_name)

#  filter probability sheet if it has node column
if 'node' in probability_sheet.columns:
    probability_sheet = probability_sheet[
        probability_sheet['node'].isin(selected_nodes)
    ]

#Save output 
with pd.ExcelWriter("reduced_output_49customers.xlsx") as writer:
    
    for sheet, df in reduced_matrices.items():
        df.to_excel(writer, sheet_name=sheet)
    
    filtered_demands.to_excel(writer, sheet_name="Filtered_Demands_49", index=False)
    probability_sheet.to_excel(writer, sheet_name="probabilities", index=False)

# Save demands separately
filtered_demands.to_excel("filtered_demands_49.xlsx", index=False)

print(" Done! Files created:")
print("- reduced_output.xlsx")
print("- filtered_demands.xlsx")