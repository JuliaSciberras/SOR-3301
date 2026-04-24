# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:10:05 2025

@author: Julia Sciberras
"""

# -*- coding: utf-8 -*-
# Full Stochastic CVRP with Scenario Travel Times (Branch-and-Cut in Gurobi)

from gurobipy import Model, GRB, quicksum
import requests
import folium
from folium.plugins import PolyLineTextPath


# Uploading the data
import pandas as pd
import numpy as np

percentiles_path = "cell_percentiles_output.xlsx"  
demand_path      = "customer_demands.xlsx"      

# Read ALL sheets into a dict: {sheet_name: DataFrame}
tt_sheets = pd.read_excel(percentiles_path, sheet_name=None, header=0, index_col=0)

# Convert to numeric NxN arrays
time_scen = {}
for scen_name, df in tt_sheets.items():
    mat = df.apply(pd.to_numeric, errors="coerce").to_numpy()
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Sheet '{scen_name}' is not square: {mat.shape}")
    time_scen[scen_name] = mat.tolist()

scenarios = list(time_scen.keys())
p = {s: 1/len(scenarios) for s in scenarios}  # for equal probability

dem_df = pd.read_excel(demand_path)

# Ensure correct columns exist
if not {"node", "demand_kg"}.issubset(dem_df.columns):
    raise ValueError("Demand file must have columns: node, demand_kg")

# Build a demand vector aligned with node numbers
dem_df["node"] = dem_df["node"].astype(int)
dem_df["demand_kg"] = pd.to_numeric(dem_df["demand_kg"], errors="coerce").fillna(0.0)

# Infer N from the travel-time matrix size (safer than guessing)
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
K_total = 4                 # max number of vehicles allowed
Kset = range(K_total)       # vehicle indices: 0..K_total-1

# Capacity and costs
Q = 1000
lambda_w = 0.167
lambda_o = 0.25
T_work = 360                # 6 hours 




# Coordinates of depot and customers (for map)
coords = {
    0: (35.895217098790866, 14.460191964374761),  # depot
    1: (35.90920069524615, 14.448575317131102),
    2: (35.89469471234125, 14.359091106257843),
    3: (35.958257259756884, 14.35850370248314),
    4: (35.898718948022385, 14.511286491012374),
    5: (35.879166612071614, 14.39987123012541),
    6: (35.85695944497671, 14.529318941365016),
    7: (35.89015021650688, 14.396768907886388),
    8: (35.90871812644858, 14.421970943790319),
    9: (35.82576508828384, 14.511967489624876),
    10: (35.84165885609389, 14.482503423596281)
}

locations = [f"Node {i}" for i in range(N)]  



# Creating the model 

m = Model("Full_VRPSTT_model")

m.setParam("Seed", 1)
m.setParam("Threads", 1)     # important for reproducibility
m.setParam("ConcurrentMIP", 1)

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

m.setParam("MIPGap", 0)
m.optimize()
print("Model status:", m.Status)


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

# Create map centered on Malta
malta_map = folium.Map(location=[35.90, 14.42], zoom_start=12)

# Add depot and customers with numeric labels
for idx in range(N):
    lat, lon = coords[idx]
    folium.Marker(
        location=[lat, lon],
        popup=f"Node {idx}",
        tooltip=f"Node {idx}",
        icon=folium.DivIcon(
            html=f"""
            <div style="font-size: 12px; colour: black; font-weight: bold; text-align: center;">
                {idx}
            </div>
            """
        )
    ).add_to(malta_map)

# OSRM function to get realistic road geometry
def get_route_coords(start, end):
    # start, end are (lat, lon)
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}"
        f"?overview=full&geometries=geojson"
    )
    r = requests.get(url)
    data = r.json()
    coords_ll = data["routes"][0]["geometry"]["coordinates"]  # list of (lon, lat)
    # convert to (lat, lon)
    return [(lat, lon) for lon, lat in coords_ll]

colors = ['red', 'blue', 'brown','purple', 'orange', 'black', 'green', 'pink', 'grey', 'maroon']

for k in Kset:
    for (i, j) in arcs[k]:
        start = coords[i]
        end   = coords[j]


        try:
            polyline_coords = get_route_coords(start, end)
        except Exception:
            polyline_coords = [start, end]

        folium.PolyLine(
            polyline_coords,
            color=colors[k % len(colors)],   
            weight=5,
            opacity=0.9,
            tooltip=f"Vehicle {k+1}"
        ).add_to(malta_map)
    


# Save map
malta_map.save("Malta_VRP_routes.html")
print("\nMap saved as 'Malta_VRP_routes.html'. Open it in a browser to see the solution.")


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
