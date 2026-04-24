# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:55:40 2026

@author: Julia Sciberras
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import time 


def load_scenarios_from_excel(scenarios_xlsx: str):

    scenarios_xlsx = str(Path(scenarios_xlsx))
    xl = pd.ExcelFile(scenarios_xlsx)

    # Identify probability sheet
    prob_sheet = "probabilities"

    # Scenario sheets = all except probability sheet
    scenario_sheets = [s for s in xl.sheet_names if s != prob_sheet]

    if not scenario_sheets:
        raise ValueError("No scenario sheets found.")

    # Read probabilities
    df_prob = pd.read_excel(scenarios_xlsx, sheet_name=prob_sheet)

    if "scenario" not in df_prob.columns or "probability" not in df_prob.columns:
        raise ValueError("Probability sheet must have columns 'scenario' and 'probability'")

    p_s = dict(zip(df_prob["scenario"], df_prob["probability"]))

    # Check probabilities sum to 1
    total_p = sum(p_s.values())

    if not np.isclose(total_p, 1.0, atol=1e-6):
        raise ValueError(
            f"Scenario probabilities must sum to 1. Found sum = {total_p:.6f}"
            )

    # Read first scenario to get node order
    first = scenario_sheets[0]
    df0 = pd.read_excel(scenarios_xlsx, sheet_name=first, index_col=0)
    nodes = list(df0.index)

    # Load matrices
    T_s = {}
    for s in scenario_sheets:
        df = pd.read_excel(scenarios_xlsx, sheet_name=s, index_col=0)

        df = df.reindex(index=nodes, columns=nodes)
        T = df.to_numpy(dtype=float)
        np.fill_diagonal(T, 0.0)

        T_s[s] = T

    return nodes, T_s, p_s

def load_demands_from_excel(
    demands_xlsx: str,
    node_col: str = "node",
    demand_col: str = "demand_kg",
) -> Dict[int, float]:

    df = pd.read_excel(demands_xlsx)
    if node_col not in df.columns or demand_col not in df.columns:
        raise ValueError(
            f"Demands file must contain columns '{node_col}' and '{demand_col}'. "
            f"Found: {list(df.columns)}"
        )
    q = {int(r[node_col]): float(r[demand_col]) for _, r in df[[node_col, demand_col]].dropna().iterrows()}
    return q


def expected_travel_time_matrix(
    T_s: Dict[str, np.ndarray],
    p_s: Dict[str, float],
) -> np.ndarray:
    """
    Computes expected travel time matrix:
      T_exp[i,j] = sum_s p_s[s] * T_s[s][i,j]
    """
    scenarios = [s for s in T_s.keys() if s in p_s]
    if not scenarios:
        raise ValueError("No overlapping scenarios between T_s and p_s.")

    T_exp = np.zeros_like(next(iter(T_s.values())))
    for s in scenarios:
        T_exp += p_s[s] * T_s[s]
    return T_exp


# Applying the model 

def route_time_in_scenario(
    route: List[int],
    T: np.ndarray,
    node_to_idx: Dict[int, int],
    m: Dict[int, float],
) -> float:
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        ia, ib = node_to_idx[a], node_to_idx[b]
        total += T[ia, ib] + m.get(b, 0.0)
    return total


def route_overtime(
    T_route: float,
    T_work: float,
) -> float:
    return max(0.0, T_route - T_work)


def expected_route_cost(
    route: List[int],
    T_s: Dict[str, np.ndarray],
    p_s: Dict[str, float],
    node_to_idx: Dict[int, int],
    m: Dict[int, float],
    T_work: float,
    lambda_w: float,
    lambda_o: float,
    dist_cost_coef: float = 0.0456,
    use_time_as_dist: bool = True,
) -> float:
   
    cost = 0.0
    for s, Tmat in T_s.items():
        ps = p_s.get(s, 0.0)
        if ps == 0.0:
            continue

        # term: sum of arc travel times (without service times)
        arc_sum = 0.0
        for a, b in zip(route[:-1], route[1:]):
            ia, ib = node_to_idx[a], node_to_idx[b]
            arc_sum += Tmat[ia, ib]

        Tks = route_time_in_scenario(route, Tmat, node_to_idx, m)
        Oks = route_overtime(Tks, T_work)

        cost += ps * (dist_cost_coef * arc_sum + lambda_w * Tks + lambda_o * Oks)

    return cost

# Starting the NN heuristic. 

def nearest_neighbor_init(
    nodes: List[int],
    T_exp: np.ndarray,
    q: Dict[int, float],
    Q: float = 1000.0,
    depot: int = 0,
) -> List[List[int]]:

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    customers = [n for n in nodes if n != depot]
    unvisited = set(customers)

    routes: List[List[int]] = []

    while unvisited: #while there are unvisited custoomers (unassigned)
        route = [] #storage for route 
        remaining = Q #Initially the remaining load is the max capacity 
        current = depot

        while True:
            # feasible candidates are unvisited customers that fit remaining capacity
            feasible = [j for j in unvisited if q.get(j, 0.0) <= remaining]
            if not feasible:
                break

            # choose the feasible customer with minimum expected travel time from current
            ic = node_to_idx[current]
            next_customer = min(
                feasible,
                key=lambda j: T_exp[ic, node_to_idx[j]]
            )

            route.append(next_customer)
            unvisited.remove(next_customer) #remove the selected customer from unvisited 
            remaining -= q.get(next_customer, 0.0)
            current = next_customer

        routes.append(route)

    return routes

def savings_init(
    nodes: List[int],
    c: np.ndarray,                 # cost matrix (e.g., expected travel time or 0.0456*expected time)
    q: Dict[int, float],
    Q: float = 1000.0,
    depot: int = 0,
) -> List[List[int]]:
    """
    Clarke–Wright Savings heuristic (capacity-feasible).
    Returns routes as lists of customers (without depot), e.g. [[3,7,2], [5,1], ...]
    """
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    customers = [n for n in nodes if n != depot]

    # Start: one route per customer
    routes = [[i] for i in customers]  # customers-only representation
    route_load = {i: q.get(i, 0.0) for i in customers}  # key by first (and only) customer initially

    # Helper: find route that starts with j / ends with i
    def find_route_starting_with(j):
        for k in routes:
            if k and k[0] == j:
                return k
        return None

    def find_route_ending_with(i):
        for k in routes:
            if k and k[-1] == i:
                return k
        return None

    # Compute savings list
    savings: List[Tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i == j:
                continue
            ii, jj, d0 = node_to_idx[i], node_to_idx[j], node_to_idx[depot]
            sav = c[ii, d0] + c[d0, jj] - c[ii, jj]
            savings.append((sav, i, j))

    # Sort descending savings
    savings.sort(reverse=True, key=lambda x: x[0])

    # Iterate merges
    for sav, i, j in savings:
        k1 = find_route_ending_with(i)
        k2 = find_route_starting_with(j)

        if k1 is None or k2 is None:
            continue
        if k1 is k2:
            continue  # prevents merging a route with itself

        load1 = sum(q.get(u, 0.0) for u in k1)
        load2 = sum(q.get(u, 0.0) for u in k2)

        if load1 + load2 <= Q:
            # merge by connecting i -> j, i at end of r1 and j at start of r2
            merged = k1 + k2
            routes.remove(k1)
            routes.remove(k2)
            routes.append(merged)

    return routes

#Lr3opt algorithm 

def expected_solution_cost(
    routes_customers: List[List[int]],
    T_s: Dict[str, np.ndarray],
    p_s: Dict[str, float],
    node_to_idx: Dict[int, int],
    m: Dict[int, float],
    T_work: float,
    lambda_w: float,
    lambda_o: float,
    depot: int = 0,
) -> float:
    total = 0.0
    for r in routes_customers:
        route = [depot] + r + [depot]
        total += expected_route_cost(
            route=route,
            T_s=T_s,
            p_s=p_s,
            node_to_idx=node_to_idx,
            m=m,
            T_work=T_work,
            lambda_w=lambda_w,
            lambda_o=lambda_o,
        )
    return total

def capacity_overload(route_customers: List[int], q: Dict[int, float], Q: float) -> float:
    load = sum(q[i] for i in route_customers)
    return max(0.0, load - Q)

def lagrangian_penalty(routes_customers: List[List[int]], q: Dict[int, float], Q: float, lam: float) -> float:
    pen = 0.00
    for r in routes_customers:
        ov = capacity_overload(r, q, Q)
        if ov > 0:
            pen += ov
    return lam * pen #lambda increases iteratively by the penalty 

#augmented objective (by introducing the lagrangian terms)
def augmented_objective(
    routes_customers: List[List[int]],
    T_s: Dict[str, np.ndarray],
    p_s: Dict[str, float],
    node_to_idx: Dict[int, int],
    m: Dict[int, float],
    T_work: float,
    lambda_w: float,
    lambda_o: float,
    q: Dict[int, float],
    Q: float,
    lam: float,
    depot: int = 0,
) -> float:
    base = expected_solution_cost(
        routes_customers, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o, depot
    )
    return base + lagrangian_penalty(routes_customers, q, Q, lam)

def first_improving_relocate(routes, q, Q, eval_fn):
    old_val = eval_fn(routes)

    for a_idx, ra in enumerate(routes):
        for pos_a, cust in enumerate(ra):
            for b_idx, rb in enumerate(routes):
                if a_idx == b_idx:
                    continue
                for pos_b in range(len(rb) + 1):

                    new_routes = [r.copy() for r in routes]
                    new_routes[a_idx].pop(pos_a)
                    new_routes[b_idx].insert(pos_b, cust)

                    new_routes = [r for r in new_routes if len(r) > 0]

                    new_val = eval_fn(new_routes)

                    # first improvement and it returns immediately
                    if new_val < old_val:
                        return new_routes, old_val - new_val

    return None, 0.0



def first_improving_swap(routes, eval_fn):
    old_val = eval_fn(routes)

    for a_idx in range(len(routes)):
        for b_idx in range(a_idx + 1, len(routes)):
            ra, rb = routes[a_idx], routes[b_idx]

            for pos_a in range(len(ra)):
                for pos_b in range(len(rb)):

                    new_routes = [r.copy() for r in routes]
                    new_routes[a_idx][pos_a], new_routes[b_idx][pos_b] = \
                        new_routes[b_idx][pos_b], new_routes[a_idx][pos_a]

                    new_val = eval_fn(new_routes)

                    # first improvement
                    if new_val < old_val:
                        return new_routes, old_val - new_val

    return None, 0.0


def lr3opt_init(
    nodes: List[int],
    q: Dict[int, float],
    Q: float,
    T_s: Dict[str, np.ndarray],
    p_s: Dict[str, float],
    node_to_idx: Dict[int, int],
    m: Dict[int, float],
    T_work: float,
    lambda_w: float,
    lambda_o: float,
    depot: int = 0,
    lam0: float = 0.05,
    lam_mult: float = 2.0,
    max_lam_updates: int = 20,
) -> List[List[int]]:
    
    start_lr3 = time.time()
    print("\nStarting LR3OPT...")


    # List of customers (exclude depot)
    customers = [n for n in nodes if n != depot]

    # Initial solution: one route per customer 
    routes = [[i] for i in customers]

    # Initial Lagrangian penalty parameter
    lam = lam0

    # Define augmented objective function (cost + penalty)
    def eval_aug(rs):
        return augmented_objective(
            rs, T_s, p_s, node_to_idx, m,
            T_work, lambda_w, lambda_o,
            q, Q, lam, depot
        )

    #Lagrangian iterations

    for it in range(max_lam_updates):
        print(f"\nLR3OPT iteration {it+1}/{max_lam_updates}, lambda = {lam:.4f}")

        #Local search phase
        move_count = 0
        # Repeat until no improving move exists
        while True:

            improved = False  # track if any move improves solution

            # Try relocate move
            # Move one customer to another route
            new_routes, _ = first_improving_relocate(routes, q, Q, eval_aug)

            if new_routes is not None:
                routes = new_routes          # accept improving move
                improved = True              # mark improvement found
                move_count += 1
                print(f" Relocate move accepted (#{move_count})")
                continue                    # restart search immediately

            # Try swap move 
            # Swap customers between two routes
            new_routes, _ = first_improving_swap(routes, eval_aug)

            if new_routes is not None:
                routes = new_routes          # accept improving move
                improved = True              # mark improvement found
                move_count += 1
                print(f"  Swap move accepted (#{move_count})")
                continue                    # restart search

            
            # stop when no improving move found 
            if not improved:
                print(f"  Local search finished after {move_count} moves")
                break

        #Check feasibility

        # Check if all routes satisfy capacity constraint
        infeasible = any(
            sum(q[i] for i in r) > Q 
            for r in routes
        )

        if not infeasible:
            print("  Feasible solution found!")
            end_lr3 = time.time()
            print(f"LR3OPT runtime: {end_lr3 - start_lr3:.2f} seconds")
            return routes

       
        # Update penalty parameter
        
        # Increase λ to penalize capacity violations more strongly
        print(" Still infeasible, thus increasing penalty")
        lam *= lam_mult

    
    # Return best found solution
    end_lr3 = time.time()
    print(f"LR3OPT finished (possibly infeasible) in {end_lr3 - start_lr3:.2f} seconds")

    # If still infeasible after all λ updates, return current solution
    return routes

def print_solution_cost(
        name,
        routes_customers,
        T_s,
        p_s,
        node_to_idx,
        m,
        T_work,
        lambda_w,
        lambda_o,
        depot=0,
        ):

    routes_with_depot = [[depot] + r + [depot] for r in routes_customers]

    total_cost = 0.0
    total_time = 0.0
    total_overtime = 0.0

    print(f"\nExpected metrics per route ({name}):")

    for rid, rt in enumerate(routes_with_depot, 1):

        # cost
        c = expected_route_cost(
            rt, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o
        )

        # time + overtime
        t, o = expected_time_and_overtime(
            rt, T_s, p_s, node_to_idx, m, T_work
        )

        total_cost += c
        total_time += t
        total_overtime += o

        print(f"  Route {rid}:")
        print(f"    cost      = {c:.4f}")
        print(f"    time      = {t:.2f} min")
        print(f"    overtime  = {o:.2f} min")

    print(f"\nTOTAL ({name}):")
    print(f"  cost      = {total_cost:.4f}")
    print(f"  time      = {total_time:.2f} min")
    print(f"  overtime  = {total_overtime:.2f} min")

def expected_time_and_overtime(
    route,
    T_s,
    p_s,
    node_to_idx,
    m,
    T_work
):
    exp_time = 0.0
    exp_overtime = 0.0

    for s, T in T_s.items():
        ps = p_s.get(s, 0.0)

        Tks = route_time_in_scenario(route, T, node_to_idx, m)
        Oks = route_overtime(Tks, T_work)

        exp_time += ps * Tks
        exp_overtime += ps * Oks

    return exp_time, exp_overtime


def collect_route_statistics(
    name,
    routes_customers,
    T_s,
    p_s,
    node_to_idx,
    m,
    T_work,
    lambda_w,
    lambda_o,
    depot=0,
):
    rows = []

    total_cost = 0.0
    total_time = 0.0
    total_overtime = 0.0

    for rid, r in enumerate(routes_customers, 1):
        route = [depot] + r + [depot]

        cost = expected_route_cost(
            route, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o
        )

        time, overtime = expected_time_and_overtime(
            route, T_s, p_s, node_to_idx, m, T_work
        )

        total_cost += cost
        total_time += time
        total_overtime += overtime

        rows.append({
            "method": name,
            "route_id": rid,
            "cost": cost,
            "time": time,
            "overtime": overtime,
        })

    # Add total row
    rows.append({
        "method": name,
        "route_id": "TOTAL",
        "cost": total_cost,
        "time": total_time,
        "overtime": total_overtime,
    })

    return rows


# Example usage
if __name__ == "__main__":
    start_time=time.time()
    scenarios_file = "scenario_input_file_for_respective_instance.xlsx"
    demands_file = "demands_input_file_for_respective_instance.xlsx"

    nodes, T_s, p_s = load_scenarios_from_excel(scenarios_file)
    print(f"Loaded {len(nodes)} nodes and {len(T_s)} scenarios")

    # Load demands
    q = load_demands_from_excel(demands_file, node_col="node", demand_col="demand_kg")
    print(f"Loaded {len(q)} customer demands")

    # Map node -> index for matrices
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Expected travel time matrix for NN
    T_exp = expected_travel_time_matrix(T_s, p_s)
    
    # cost matrix for savings
    c_matrix = T_exp
    
    routes_savings = savings_init(nodes=nodes, c=c_matrix, q=q, Q=1000.0, depot=0)

    print("\nSavings initial routes (customers only):")
    for k_id, k in enumerate(routes_savings, start=1):
        load = sum(q[i] for i in k)
        print(f"  Route {k_id}: {k} | load={load:.2f} kg")

    # Service times: customers 10, depot 0
    m = {node: 10.0 for node in nodes}
    m[0] = 0.0

    # NN initialization
    Q = 1000.0
    routes = nearest_neighbor_init(nodes=nodes, T_exp=T_exp, q=q, Q=Q, depot=0)

    print("\nNN initial routes (customers only):")
    for r_id, r in enumerate(routes, start=1):
        load = sum(q[i] for i in r)
        print(f"  Route {r_id}: {r} | load={load:.2f} kg")

    # Add depot for evaluation
    routes_with_depot = [[0] + r + [0] for r in routes]

    # Parameters for expected cost evaluation
    T_work = 6 * 60.0
    lambda_w = 0.167
    lambda_o = 0.25
    
    routes_lr3 = lr3opt_init(
    nodes=nodes,
    q=q,
    Q=1000.0,
    T_s=T_s,
    p_s=p_s,
    node_to_idx=node_to_idx,
    m=m,
    T_work=T_work,
    lambda_w=lambda_w,
    lambda_o=lambda_o,
    depot=0,
    lam0=0.05,
    lam_mult=2.0,
    max_lam_updates=15,
)
    print_solution_cost(
    "NN",
    routes,
    T_s, p_s, node_to_idx, m,
    T_work, lambda_w, lambda_o
    )

    print_solution_cost(
    "Savings",
    routes_savings,
    T_s, p_s, node_to_idx, m,
    T_work, lambda_w, lambda_o
)

    print_solution_cost(
    "LR3OPT",
    routes_lr3,
    T_s, p_s, node_to_idx, m,
    T_work, lambda_w, lambda_o
)


    print("\nLR3OPT initial routes (customers only):")
    for r_id, r in enumerate(routes_lr3, start=1):
        load = sum(q[i] for i in r)
        print(f"  Route {r_id}: {r} | load={load:.2f} kg")


    rows_nn = []
    for r_id, r in enumerate(routes, start=1):
        for pos, cust in enumerate(r, start=1):
            rows_nn.append({
                "route_id": r_id,
                "position": pos,
                "customer": cust,
                "demand": q[cust]
                })

    rows_sav = []
    for r_id, r in enumerate(routes_savings, start=1):
        for pos, cust in enumerate(r, start=1):
            rows_sav.append({
                "route_id": r_id,
                "position": pos,
                "customer": cust,
                "demand": q[cust]
                })
            
    rows_lr3 = []
    for r_id, r in enumerate(routes_lr3, start=1):
        for pos, cust in enumerate(r, start=1):
            rows_lr3.append({"route_id": r_id, "position": pos, "customer": cust, "demand": q[cust]})
            
    rows_stats = []

    rows_stats += collect_route_statistics(
        "NN", routes, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o
        )

    rows_stats += collect_route_statistics(
        "Savings", routes_savings, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o
        )

    rows_stats += collect_route_statistics(
        "LR3OPT", routes_lr3, T_s, p_s, node_to_idx, m, T_work, lambda_w, lambda_o
        )
    
    end_time=time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


    with pd.ExcelWriter("all_initial_solutions_149customers.xlsx") as writer:
        pd.DataFrame(rows_nn).to_excel(writer, sheet_name="NN", index=False)
        pd.DataFrame(rows_sav).to_excel(writer, sheet_name="Savings", index=False)
        pd.DataFrame(rows_lr3).to_excel(writer, sheet_name="LR3OPT", index=False)
        pd.DataFrame(rows_stats).to_excel(writer, sheet_name="Summary", index=False)


    print("\nSaved initial_solutions.xlsx (NN + Savings+LR3opt)")



