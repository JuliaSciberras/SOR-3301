# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:35:26 2026

@author: Julia Sciberras
"""

from gurobipy import Model, GRB, quicksum

# Uploading the data
import pandas as pd
import numpy as np
import sys

percentiles_path = "Example_percentiles_input_Sheet.xlsx"  
demand_path      = "demands_input_sheet.xlsx"      

# Read all sheets in workbook
all_sheets = pd.read_excel(percentiles_path, sheet_name=None, header=0, index_col=0)

# Keep only scenario matrix sheets
tt_sheets = {name: df for name, df in all_sheets.items() if name != "probabilities"}

# Read probability sheet separately
prob_df = pd.read_excel(percentiles_path, sheet_name="probabilities")

# Convert scenario sheets to numeric NxN matrices
time_scen = {}

for scen_name, df in tt_sheets.items():
    mat = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Sheet '{scen_name}' is not square: {mat.shape}")

    time_scen[scen_name] = mat.tolist()

# Scenario names
scenarios = list(time_scen.keys())

# Probabilities from Excel
p = dict(zip(prob_df["scenario"], prob_df["probability"]))

# Checks
missing_probs = [s for s in scenarios if s not in p]
if missing_probs:
    raise ValueError(f"Missing probabilities for: {missing_probs}")

if abs(sum(p.values()) - 1) > 1e-6:
    raise ValueError("Probabilities must sum to 1.")

dem_df = pd.read_excel(demand_path)

# Ensure correct columns exist
if not {"node", "demand_kg"}.issubset(dem_df.columns):
    raise ValueError("Demand file must have columns: node, demand_kg")

# Build a demand vector aligned with node numbers
dem_df["node"] = dem_df["node"].astype(int)
dem_df["demand_kg"] = pd.to_numeric(dem_df["demand_kg"], errors="coerce").fillna(0.0)

# Infer N from the travel-time matrix size
N = next(iter(time_scen.values())).__len__()  # size of first matrix

demand = [0.0] * N
for _, r in dem_df.iterrows():
    node = int(r["node"])
    if 0 <= node < N:
        demand[node] = float(r["demand_kg"])

# Optional sanity check:
missing = [i for i in range(N) if i not in set(dem_df["node"])]
# (you can ignore missing nodes or raise an error)


V = range(N)
V_c = range(1, N)

# Vehicles
K_total = 38                 # max number of vehicles allowed (Set for the 150 locations instance)
Kset = range(K_total)       # vehicle indices: 0..K_total-1

# Capacity and costs
Q = 1000
lambda_w = 0.167
lambda_o = 0.25
T_work = 360                



locations = [f"Node {i}" for i in range(N)]  



# Creating the model 

m = Model("Full_VRPSTT_model")

m.setParam("Seed", 1)
m.setParam("Threads", 0)     # important for reproducibility


# Variables
x = m.addVars(V, V, Kset, vtype=GRB.BINARY, name="x")         # arc usage
y = m.addVars(Kset, vtype=GRB.BINARY, name="y")               # vehicle used
u = m.addVars(V_c, Kset, lb=0, ub=Q, name="u")                # load at customer
T = m.addVars(Kset, scenarios, lb=0, name="T")                # total time per vehicle & scenario
O = m.addVars(Kset, scenarios, lb=0, ub=60, name="O")         # overtime per vehicle & scenario


# Objective function 

m.setObjective(
    quicksum(
        0.0456 * p[s] * time_scen[s][i][j] * x[i, j, k]
        for s in scenarios
        for i in V for j in V if i != j
        for k in Kset
    )
    + lambda_w * quicksum(
        p[s] * quicksum(T[k, s] for k in Kset)
        for s in scenarios
    )
    + lambda_o * quicksum(
        p[s] * quicksum(O[k, s] for k in Kset)
        for s in scenarios
    ),
    GRB.MINIMIZE
)


# Constraints (3.2)–(3.12)


# (3.2) each customer has exactly one incoming arc across all vehicles
for j in V_c:
    m.addConstr(quicksum(x[i, j, k] for i in V if i != j for k in Kset) == 1,
                name=f"visit_{j}")

# (3.3) Flow conservation per vehicle:
# for each vehicle k and each node j>0: inflow = outflow
for k in Kset:
    for h in V_c:  # node where flow must balance
        m.addConstr(
            quicksum(x[i, h, k] for i in V if i != h) ==
            quicksum(x[h, j, k] for j in V if j != h),
            name=f"flow[{k},{h}]"
        )


# (3.4) start depot: number of departures from depot = y_k
for k in Kset:
    m.addConstr(quicksum(x[0, j, k] for j in V if j != 0) == y[k],
                name=f"start_depot[{k}]")

# (3.5) end depot: number of arrivals to depot = y_k
for k in Kset:
    m.addConstr(quicksum(x[i, 0, k] for i in V if i != 0) == y[k],
                name=f"end_depot[{k}]")

# (3.6) total vehicles used ≤ K_total
m.addConstr(quicksum(y[k] for k in Kset) <= K_total, name="num_vehicles")

# (3.7)–(3.8) Load balance inequalities (MTZ-style)
for k in Kset:
    for i in V_c:
        for j in V_c:
            if i != j:
                m.addConstr(
                    u[j, k] <= u[i, k] - demand[j] + Q * (1 - x[i, j, k]),
                    name=f"load_up[{i},{j},{k}]"
                )
                m.addConstr(
                    u[j, k] >= u[i, k] - demand[j] - Q * (1 - x[i, j, k]),
                    name=f"load_lo[{i},{j},{k}]"
                )

# (3.9) 0 ≤ u_i^k ≤ Q*y_k  (capacity scaling with vehicle usage)
for k in Kset:
    for i in V_c:
        m.addConstr(u[i, k] <= Q * y[k], name=f"cap_up[{i},{k}]")
        m.addConstr(u[i, k] >= 0,        name=f"cap_lo[{i},{k}]")

# (3.10) Travel time definition per scenario:
# travel time + service time at customers
m_j = {i: (0 if i == 0 else 10) for i in V}
for s in scenarios:
    for k in Kset:
        m.addConstr(
            T[k, s] == quicksum(
                (time_scen[s][i][j] + m_j[j]) * x[i, j, k]
                for i in V for j in V if i != j
            ),
            name=f"time[{k},{s}]"
        )


# (3.11) Overtime definition
for s in scenarios:
    for k in Kset:
        m.addConstr(O[k, s] >= T[k, s] - T_work,
                    name=f"overtime[{k},{s}]")

# (3.12) O_k^s upper bound already handled in variable definition (ub=60)
# (3.13)-(3.14) domain handled by variable types



# Solving the problem 

# Solving the problem

m.setParam("MIPGap", 0)          # require proven optimality if found
m.setParam("TimeLimit", 14400)   # 4 hour time limit 

# Optional speed improvements
m.setParam("Heuristics", 0.3)
m.setParam("Cuts", 2)
m.setParam("Presolve", 2)

m.optimize()

status = m.Status
print("Model status:", status)

if status == GRB.OPTIMAL:
    print("\nOptimal solution found.")

elif status == GRB.TIME_LIMIT:
    print("\nTime limit reached.")
    print("Returning best feasible solution found.")

elif status == GRB.INFEASIBLE:
    print("\nModel is infeasible.")
    sys.exit()

else:
    print(f"\nSolver ended with status code {status}")

# Check if any feasible solution exists
if m.SolCount == 0:
    print("\nNo feasible solution found.")
    sys.exit()

# Report quality of best solution
print(f"\nBest objective value = {m.ObjVal:.4f}")
print(f"Best bound           = {m.ObjBound:.4f}")
print(f"Optimality gap       = {100*m.MIPGap:.2f}%")


# Display routes and times 


def extract_route(k, x, N, start=0):
    route = [start]
    cur = start
    visited = {start}
    while True:
        nxt = None
        for j in range(N):
            if j != cur and x[cur, j, k].X > 0.5:
                nxt = j
                break
        if nxt is None:
            break
        route.append(nxt)
        if nxt == start:
            break
        if nxt in visited:  # safety against cycles
            break
        visited.add(nxt)
        cur = nxt
    return route

print("\n--- Routes ---")
for k in Kset:
    if y[k].X < 0.5:
        continue
    r = extract_route(k, x, N, start=0)
    print(f"Vehicle {k+1}: " + " → ".join(locations[i] for i in r))

print("\n--- Route completion times ---")
for s in scenarios:
    for k in Kset:
        if y[k].X > 0.5:
            print(f"Vehicle {k+1}, {s} traffic:")
            print(f"  Total time T = {T[k, s].X:.2f} min")
            print(f"  Overtime  O = {O[k, s].X:.2f} min")


# Creating the map of Malta with routes showing 


# Extract used arcs from the optimal solution
arcs = {k: [] for k in Kset}
for k in Kset:
    for i in V:
        for j in V:
            if i != j and x[i, j, k].X > 0.5:
                arcs[k].append((i, j))



print("\n=== Results Summary ===")

# Times
print("\n--- Vehicle Times (T_k^s) ---")
for s in scenarios:
    for k in Kset:
        if y[k].X > 0.5:
            print(f"Scenario {s}, Vehicle {k+1}: T = {T[k,s].X:.2f} min")

# Overtime
print("\n--- Overtime (O_k^s) ---")
for s in scenarios:
    for k in Kset:
        if y[k].X > 0.5:
            print(f"Scenario {s}, Vehicle {k+1}: O = {O[k,s].X:.2f} min")

#Loads 
print("\n--- Vehicle Remaining Load (computed from route) ---")
for k in Kset:
    if y[k].X < 0.5:
        continue

    r = extract_route(k, x, N, start=0)

    rem = Q
    print(f"\nVehicle {k+1}:")
    for node in r[1:]:              # skip the starting depot
        if node == 0:
            print(f"  Return depot: remaining load = {rem:.0f}")
            break
        rem -= demand[node]
        print(f"  After visiting {locations[node]}: remaining load = {rem:.0f}")


# Total load
print("\n--- Total load delivered by each vehicle ---")
for k in Kset:
    if y[k].X > 0.5:
        load_k = sum(demand[j] for (i,j) in arcs[k] if j != 0)
        print(f"Vehicle {k+1}: total load = {load_k}")
