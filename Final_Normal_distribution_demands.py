# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:11:31 2026

@author: Julia Sciberras
"""

import numpy as np
import pandas as pd

def generate_demands_truncated_normal(
    N_customers: int,
    Q: float,
    mu: float,
    sigma: float,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    x = rng.normal(mu, sigma, size=N_customers) #normal distribution 
    mask = (x < 0) | (x > Q)

    while mask.any():
        x[mask] = rng.normal(mu, sigma, size=mask.sum())
        mask = (x < 0) | (x > Q)

    q = np.zeros(N_customers + 1)
    q[1:] = x
    q[0] = 0.0
    return q

# parameters
N = 149
Q = 1000.0
mu = 100 #Adjust according the instance size 
sigma = 35 #Adjust according the instance size 

q = generate_demands_truncated_normal(N, Q, mu, sigma, seed=42)

df_demands = pd.DataFrame({
    "node": list(range(N + 1)),
    "demand_kg": q
})

df_demands.to_excel("customer_demands_for149customers.xlsx", index=False)



