# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:02:27 2026

@author: Julia Sciberras
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load file
file = "ALNS_final_solution_149customers_Savings.xlsx"
df = pd.read_excel(file, sheet_name="Operator_Log")

# Convert booleans
for col in ['accepted','improved','new_best']:
    df[col] = df[col].astype(str).str.upper() == 'TRUE'

# Metrics
# overall acceptance rate
overall = df['accepted'].mean()

# worse proposals
worse = df['f_candidate'] > df['f_current']

# accepted worse proposals
worse_accept = (worse & df['accepted']).sum() / worse.sum()

print("Overall acceptance:", overall)
print("Worse-solution acceptance:", worse_accept)
imp_rate = df['improved'].mean()
best_rate = df['new_best'].mean()

print("worse_accept:", round(worse_accept,3))
print("Improvement Rate:", round(imp_rate,3))
print("New Best Rate:", round(best_rate,3))

# Rolling rates
window = 50
df['acc_roll'] = df['accepted'].rolling(window).mean()
df['imp_roll'] = df['improved'].rolling(window).mean()

# --- Graphs ---
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.plot(df['iteration'], df['f_current'], label='Current')
plt.plot(df['iteration'], df['f_best_after'], label='Best')
plt.title("Objective Values")
plt.legend()

plt.subplot(2,2,2)
plt.plot(df['iteration'], df['acc_roll'], label='Acceptance')
plt.plot(df['iteration'], df['imp_roll'], label='Improvement')
plt.title("Rolling Rates")
plt.legend()

plt.subplot(2,2,3)
df['destroy_op'].value_counts().plot(kind='bar')
plt.title("Destroy Operator Usage")

plt.subplot(2,2,4)
df['repair_op'].value_counts().plot(kind='bar')
plt.title("Repair Operator Usage")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))

plt.plot(df['f_current'],
         label='Current Solution',
         linewidth=2.5,
         color='blue')

plt.plot(df['f_candidate'],
         label='Candidate Solution',
         linewidth=1.8,
         color='orange',
         alpha=0.8,
         linestyle='--')

plt.title("Current vs Candidate Solution Quality")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()