# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:44:16 2026

@author: Julia Sciberras
"""

import pandas as pd
import numpy as np

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load matrices from Excel
combined_file = "all_travel_times_combined.xlsx"

xls = pd.ExcelFile(combined_file)
sheet_names = xls.sheet_names
print("Found sheets:", sheet_names)

matrices = []

for sheet in sheet_names:
    print(f"Reading sheet: {sheet}")

    # Read sheet
    df = pd.read_excel(xls, sheet_name=sheet, header=0)

    df = pd.read_excel(xls, sheet_name=sheet, header=0, index_col=0)
    numeric_df = df
    # Convert to numeric
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

    # Check shape (should be 149x149)
    if numeric_df.shape != (150, 150):
        print(f"WARNING: {sheet} -> shape {numeric_df.shape}, expected (149, 149)")

    M = numeric_df.to_numpy(dtype=float)
    matrices.append(M)

print("Number of matrices loaded:", len(matrices))
n_obs = len(matrices)

# Flatten matrices
def flatten_matrix(M: np.ndarray) -> np.ndarray: #to create a vector shape 
    return M.reshape(-1)

X = np.vstack([flatten_matrix(M) for M in matrices])
print("X shape:", X.shape)


# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
npc = np.searchsorted(cum_var, 0.95) + 1 #finding minimum number of PCS explained greater or equal to variance 

print("Cumulative variance:", cum_var)
print(f"Using {npc} PCs, explaining {cum_var[npc-1]:.3f} of variance")

X_pca = X_pca_full[:, :npc]


# KMeans + silhouette
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_values = range(3, 11) #trying clusters from 3 to 10 
results = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        n_init=50, #run 50 times and pick best solution 
        random_state=RANDOM_SEED
    )
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}, silhouette={score:.3f}")
    results.append((k, score))

best_k, best_score = max(results, key=lambda t: t[1])
print("\nBest configuration:")
print(f"  k (scenarios) = {best_k}")
print(f"  silhouette = {best_score:.3f}")


# Final clustering
final_kmeans = KMeans(
    n_clusters=best_k,
    n_init=100,
    random_state=RANDOM_SEED
)
labels = final_kmeans.fit_predict(X_pca)


# Silhouette plot
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

def plot_silhouette(X_pca, labels, title="Silhouette Plot"):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X_pca, labels)

    y_lower = 10
    plt.figure(figsize=(8, 6))

    for c in range(n_clusters):
        cluster_sil_vals = silhouette_vals[labels == c]
        cluster_sil_vals.sort()

        y_upper = y_lower + len(cluster_sil_vals)
        color = plt.cm.nipy_spectral(float(c) / n_clusters)

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_sil_vals,
            facecolor=color, edgecolor=color, alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(c))
        y_lower = y_upper + 10

    silhouette_avg = silhouette_score(X_pca, labels)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster Label")
    plt.title(f"{title}\nAverage Silhouette = {silhouette_avg:.3f}")
    plt.show()

plot_silhouette(X_pca, labels, title=f"Silhouette Plot: {best_k} Scenarios (KMeans)")


# Build scenario matrices
scenario_mats = []
scenario_probs = []

for c in range(best_k):
    idx = np.where(labels == c)[0]

    if len(idx) == 0:
        continue  # safety

    prob = len(idx) / n_obs
    scenario_probs.append(prob)

    cluster_mats = [matrices[i] for i in idx]
    scenario_mat = np.mean(cluster_mats, axis=0)
    scenario_mats.append(scenario_mat)

print("Scenario probabilities:", scenario_probs)
print("Number of scenarios:", len(scenario_mats))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.axvline(x=npc, color='green', linestyle='--', label=f'{npc} PCs')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()


# Save to Excel
output_scenarios_file = "scenario_matrices_kmeans.xlsx"

if len(scenario_mats) == 0:
    raise ValueError("No scenario matrices to save.")

with pd.ExcelWriter(output_scenarios_file, engine="openpyxl") as writer:

    # Write scenario matrices
    for s_idx, mat in enumerate(scenario_mats, start=1):
        n = mat.shape[0]  # dynamic size (149)
        df_out = pd.DataFrame(mat, index=range(n), columns=range(n))
        df_out.to_excel(writer, sheet_name=f"scenario_{s_idx}")

    # Write probabilities
    prob_df = pd.DataFrame({
        "scenario": [f"scenario_{i+1}" for i in range(len(scenario_mats))],
        "probability": scenario_probs
    })
    prob_df.to_excel(writer, sheet_name="probabilities", index=False)

print(f"Saved scenario matrices to: {output_scenarios_file}")

print("\nSilhouette Scores:")
for k, score in results:
    print(f"k={k}: {score:.4f}")

print(f"\nBest k = {best_k}")
print(f"Best silhouette = {best_score:.4f}")

print("\nScenario Probabilities:")
for i, p in enumerate(scenario_probs, start=1):
    print(f"Scenario {i}: {p:.4f}")