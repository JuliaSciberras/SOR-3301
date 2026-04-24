# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:45:58 2026

@author: Julia Sciberras
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import time


# Data structures
@dataclass
class RemovalResult:
    S_minus_df: pd.DataFrame
    removed_df: pd.DataFrame
    alpha: float
    q: int
    chosen_route_id: int

# Reading the excel file.
def read_solution_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"route_id", "position", "customer", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' is missing columns: {missing}")

    df["route_id"] = df["route_id"].astype(int)
    df["position"] = df["position"].astype(int)
    df["customer"] = df["customer"].astype(int)

    df = df.sort_values(["route_id", "position"]).reset_index(drop=True)
    return df


# Building the routes from the excel file 
def df_to_routes(df: pd.DataFrame) -> Dict[int, List[int]]:
   
    routes: Dict[int, List[int]] = {}
    for rid, grp in df.groupby("route_id"):
        routes[int(rid)] = grp.sort_values("position")["customer"].tolist()
    return routes


def routes_to_df(routes: Dict[int, List[int]], demand_map: Dict[int, float]) -> pd.DataFrame:
   
    rows = []
    for rid in sorted(routes.keys()):
        for pos, cust in enumerate(routes[rid], start=1):
            rows.append({
                "route_id": rid,
                "position": pos,
                "customer": cust,
                "demand": float(demand_map.get(cust, np.nan))
            })
    return pd.DataFrame(rows, columns=["route_id", "position", "customer", "demand"])


#Getting the demand for insertion operators
def build_demand_map(df: pd.DataFrame) -> Dict[int, float]:
   
    dm = {}
    for _, r in df[["customer", "demand"]].drop_duplicates().iterrows():
        dm[int(r["customer"])] = float(r["demand"])
    return dm

#Loading the excel file with the scenarios of the travelling times. 

def load_travel_times_with_probabilities(excel_path: str):
    sheets = pd.read_excel(excel_path, sheet_name=None)

    sheet_names = list(sheets.keys())

    # Last sheet = probabilities
    prob_sheet_name = sheet_names[-1] #the probability is the last sheet 
    prob_df = sheets[prob_sheet_name].copy()

    # Clean column names
    prob_df.columns = [c.strip().lower() for c in prob_df.columns]

    if not {"scenario", "probability"}.issubset(prob_df.columns):
        raise ValueError("Probability sheet must contain 'scenario' and 'probability' columns.")

    probabilities = dict(zip(prob_df["scenario"], prob_df["probability"]))

    # Remaining sheets = travel times
    tt_by_scenario = {} 

    for sname in sheet_names[:-1]:
        df = sheets[sname].copy()

        df = df.set_index(df.columns[0])
        df.index = df.index.astype(int)
        df.columns = [int(c) for c in df.columns]

        tt_by_scenario[sname] = df

    # Check consistency
    for s in tt_by_scenario:
        if s not in probabilities:
            raise ValueError(f"Missing probability for scenario '{s}'.")

    return tt_by_scenario, probabilities


def expected_travel_time_matrix(tt_by_scenario, probabilities):
    scenarios = list(tt_by_scenario.keys())
    if not scenarios:
        raise ValueError("No scenario travel-time matrices provided.")

    bar_t = None

    for s in scenarios:
        p = probabilities[s]
        mat = tt_by_scenario[s].astype(float)

        if bar_t is None:
            bar_t = p * mat
        else:
            bar_t = bar_t.add(p * mat, fill_value=0.0)

    return bar_t


def route_of_customer(routes: Dict[int, List[int]], cust: int) -> Optional[int]:

    for rid, seq in routes.items():
        if cust in seq:
            return rid
    return None


# l_ij is taken to be -1 if they are same route, else +1
def l_ij_same_route(routes: Dict[int, List[int]], i: int, j: int) -> int:
    
    ri = route_of_customer(routes, i)
    rj = route_of_customer(routes, j)
    if ri is not None and ri == rj:
        return -1
    return 1




# Random removal operator (single-route)

def random_removal_single_route(
    df: pd.DataFrame,
    alpha_low: float = 0.1,
    alpha_high: float = 0.15,
    rng: Optional[np.random.Generator] = None
) -> RemovalResult:
    
    #if the random seed is not specified and then produce it randomly 
    if rng is None: 
        rng = np.random.default_rng()

    df = df.copy()
    routes = df_to_routes(df)
    demand_map = build_demand_map(df)

    # total number of customers n (unique customers in this solution)
    customers_all = df["customer"].unique().tolist()
    n = len(customers_all) #n is the number of customers 
    if n == 0:
        raise ValueError("No customers found in the solution dataframe.")

    alpha = rng.uniform(alpha_low, alpha_high) #alpha is randomly generated 
    q = int(math.ceil(alpha * n)) #q is the number of customers to be removed 

    # choose a non-empty route uniformly
    non_empty_routes = [rid for rid, seq in routes.items() if len(seq) > 0]
    if not non_empty_routes:
        raise ValueError("All routes are empty; nothing to remove.")

    chosen_route_id = int(rng.choice(non_empty_routes))
    route_seq = routes[chosen_route_id]

    q = min(q, len(route_seq)) #for this operator, you cannot remove more customers than the chosen route has 

    # choose D uniformly without replacement
    removed_customers = rng.choice(route_seq, size=q, replace=False).tolist()
    removed_set: Set[int] = set(int(x) for x in removed_customers)

    # remove the chosen customers 
    new_seq = [c for c in route_seq if c not in removed_set]
    routes[chosen_route_id] = new_seq

    # Build S^- dataframe
    S_minus_df = routes_to_df(routes, demand_map)

    # Build removed_df with original positions (from original df)
    removed_df = (
        df[df["route_id"].eq(chosen_route_id) & df["customer"].isin(removed_set)]
        .sort_values("position")
        .reset_index(drop=True)
    )

    return RemovalResult(
        S_minus_df=S_minus_df,
        removed_df=removed_df,
        alpha=float(alpha),
        q=int(q),
        chosen_route_id=chosen_route_id
    )

# Shaw removal operator 
def shaw_removal_seed_based(
    df: pd.DataFrame,
    bar_t: pd.DataFrame,                      # expected travel time matrix \bar{t}
    alpha_low: float = 0.1,
    alpha_high: float = 0.15,
    phi1: float = 1/3,
    phi2: float = 1/3,
    phi3: float = 1/3,
    rng: Optional[np.random.Generator] = None
) -> RemovalResult:
    

    if rng is None:
        rng = np.random.default_rng()

    df = df.copy()
    demand_map = build_demand_map(df)
    routes = df_to_routes(df)

    customers = df["customer"].unique().tolist()
    n = len(customers)   #n is the length of customers, i.e. total number of customers
    if n == 0:
        raise ValueError("No customers found in the solution dataframe.")

    alpha = rng.uniform(alpha_low, alpha_high)   # Alpha sampeled uniformly
    q = int(math.ceil(alpha * n))
    q = min(q, n)  # can't remove more than existing customers

    # pick seed to choose which customer should be removed 
    seed = int(rng.choice(customers))

    # Build normalization constants (denominators)
    # Tmax = max_{u,v in Vc} \bar{t}_{uv}
    # Qmax = max_{u,v in Vc} |q_u - q_v|
    # eps is there to avoid division by 0 
    eps = 1e-12

    # Tmax using only customers (not depot)
    cust_idx = [c for c in customers if c in bar_t.index and c in bar_t.columns]
    if len(cust_idx) == 0:
        raise ValueError("Customer IDs do not match indices/columns of bar_t matrix.")

    Tmax = float(bar_t.loc[cust_idx, cust_idx].to_numpy().max())
    Tmax = max(Tmax, eps)

    q_vals = [float(demand_map[c]) for c in customers if c in demand_map]
    Qmax = float(max(q_vals) - min(q_vals)) if q_vals else 0.0
    Qmax = max(Qmax, eps)

    # Compute relatedness to seed for every other customer 
    rel_scores = []
    for j in customers: #compute for all the customers, obviously not the seed custmer 
        if j == seed:
            continue

        # term 1: normalized expected travel time
        tij = float(bar_t.loc[seed, j])
        term1 = tij / Tmax

        # term 2: same-route indicator l_ij in {-1, +1}
        lij = l_ij_same_route(routes, seed, j)
        term2 = float(lij)

        # term 3: normalized demand difference
        term3 = abs(float(demand_map[seed]) - float(demand_map[j])) / Qmax

        Rel = phi1 * term1 + phi2 * term2 + phi3 * term3
        rel_scores.append((j, Rel))

    # Sort ascending: smallest Rel = most related
    rel_scores.sort(key=lambda x: x[1])

    # Removed set D includes seed + (q-1) most related
    remove_count_more = max(0, q - 1)
    chosen_more = [int(j) for j, _ in rel_scores[:remove_count_more]]
    removed_set = set([seed] + chosen_more)

    # Create removed_df from original df (keeps route_id and original position)
    removed_df = (
        df[df["customer"].isin(removed_set)]
        .sort_values(["route_id", "position"])
        .reset_index(drop=True)
    )

    # Remove them from solution and renumber positions
    S_minus_df = df[~df["customer"].isin(removed_set)].copy()
    S_minus_df = S_minus_df.sort_values(["route_id", "position"]).reset_index(drop=True)
    S_minus_df["position"] = S_minus_df.groupby("route_id").cumcount() + 1

    return RemovalResult(
        S_minus_df=S_minus_df,
        removed_df=removed_df,
        alpha=float(alpha),
        q=int(q),
        chosen_route_id=-1
    )


# The user chooses which initial solution he wants to use. (Normally, the one that produced the least initial cost is chosen.)
def choose_initial_solution_interactive(excel_path: str, default_sheet: str = "LR3OPT"):
    # Read all sheet names
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names

    # Ignore last sheet
    usable_sheets = sheets[:-1]

    print(f"Available initial solutions: {usable_sheets}")
    choice = input(f"Choose a sheet name (default = {default_sheet}): ").strip()

    if choice == "":
        choice = default_sheet

    if choice not in usable_sheets:
        raise ValueError(f"Invalid choice '{choice}'. Must be one of {usable_sheets}.")

    df = read_solution_sheet(excel_path, choice)
    return choice, df

#Specifying the objective function so it calculates the required costs. 

def objective_cost_with_overtime(
    solution_df: pd.DataFrame,
    tt_by_scenario: dict[str, pd.DataFrame],
    probabilities: dict[str, float],
    T_work: float = 360.0,          
    lambda_w: float = 0.167,
    lambda_o: float = 0.25,
    service_time_customer: float = 10.0,
    overtime_cap: float = 60.0
) -> float:
    
    # Build routes from the solution
    routes = {
        int(rid): grp.sort_values("position")["customer"].tolist()
        for rid, grp in solution_df.groupby("route_id")
    }

    scenarios = list(tt_by_scenario.keys())
    if not scenarios:
        raise ValueError("No scenario travel-time matrices provided.")


    total = 0.0

    for s in scenarios:
        tt = tt_by_scenario[s]
        p = probabilities[s]

        travel_sum = 0.0
        T_sum = 0.0
        O_sum = 0.0

        for seq in routes.values():
            if not seq:
                continue

            # travel (minutes), including depot arcs
            travel = float(tt.loc[0, seq[0]])
            for a, b in zip(seq[:-1], seq[1:]):
                travel += float(tt.loc[a, b])
            travel += float(tt.loc[seq[-1], 0])

            # service time (minutes): 10 per customer, depot has 0 and isn't in seq
            service = service_time_customer * len(seq)

            # route duration
            T_k = travel + service

            # overtime (cap at 60 like your constraints)
            O_k = max(0.0, T_k - T_work)
            if overtime_cap is not None:
                O_k = min(O_k, overtime_cap)

            travel_sum += travel
            T_sum += T_k
            O_sum += O_k

        total += p * (0.0456 * travel_sum + lambda_w * T_sum + lambda_o * O_sum)

    return total

def remove_customer_from_df(df: pd.DataFrame, customer_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    removed_rows = df[df["customer"] == int(customer_id)].copy()
    new_df = df[df["customer"] != int(customer_id)].copy()

    new_df = new_df.sort_values(["route_id", "position"]).reset_index(drop=True)
    new_df["position"] = new_df.groupby("route_id").cumcount() + 1
    return new_df, removed_rows


def worst_cost_removal_updated(
    df: pd.DataFrame,
    cost_fn,
    alpha_low: float = 0.1,
    alpha_high: float = 0.15,
    rng: Optional[np.random.Generator] = None
) -> RemovalResult:
    if rng is None:
        rng = np.random.default_rng()

    S_minus = df.copy()

    n = S_minus["customer"].nunique()
    if n == 0:
        raise ValueError("No customers found in the solution dataframe.")

    alpha = rng.uniform(alpha_low, alpha_high)
    q = int(math.ceil(alpha * n))
    q = min(q, n)

    removed_list = [] #storage for D (unassigned customers)

    for _ in range(q):
        current_customers = S_minus["customer"].unique().tolist()
        if not current_customers:
            break

        base_cost = cost_fn(S_minus)

        best_customer = None
        best_sigma = -float("inf") #setting sigma as minus infinity 

        for j in current_customers:
            S_wo_j, _ = remove_customer_from_df(S_minus, j)
            sigma_j = base_cost - cost_fn(S_wo_j) #calculating sigma 
            if sigma_j > best_sigma: #keep the hughest sigma 
                best_sigma = sigma_j
                best_customer = j

        S_minus, removed_rows = remove_customer_from_df(S_minus, best_customer)
        if len(removed_rows) > 0:
            removed_list.append(removed_rows)

    removed_df = pd.concat(removed_list, ignore_index=True) if removed_list else pd.DataFrame(columns=df.columns)

    return RemovalResult(
        S_minus_df=S_minus,
        removed_df=removed_df,
        alpha=float(alpha),
        q=int(q),
        chosen_route_id=-1
    )

def route_cost(seq, tt_by_scenario, probabilities, T_work, lambda_w, lambda_o, service_time_customer, overtime_cap):
    total = 0.0

    for s, tt in tt_by_scenario.items():
        p = probabilities[s]

        if not seq:
            continue

        travel = float(tt.loc[0, seq[0]])
        for a, b in zip(seq[:-1], seq[1:]):
            travel += float(tt.loc[a, b])
        travel += float(tt.loc[seq[-1], 0])

        service = service_time_customer * len(seq)
        T_k = travel + service

        O_k = max(0.0, T_k - T_work)
        if overtime_cap is not None:
            O_k = min(O_k, overtime_cap)

        total += p * (0.0456 * travel + lambda_w * T_k + lambda_o * O_k)

    return total


# Insertion operators start 

#Checking the feasibility of route times is important in the insertion part 
def is_route_time_feasible(
    route_seq: List[int],
    tt_by_scenario: dict[str, pd.DataFrame],
    T_work: float,
    overtime_cap: float,
    service_time_customer: float
) -> bool:

    for s, tt in tt_by_scenario.items():

        if len(route_seq) == 0:
            return True

        travel = float(tt.loc[0, route_seq[0]])

        for a, b in zip(route_seq[:-1], route_seq[1:]):
            travel += float(tt.loc[a, b])

        travel += float(tt.loc[route_seq[-1], 0])

        service = service_time_customer * len(route_seq)
        T_k = travel + service

        if T_k - T_work > overtime_cap:
            return False

    return True


# Sequential Greedy Insertion Operator 
def sequential_greedy_insertion(
    S_destroyed_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    tt_by_scenario: dict[str, pd.DataFrame],
    Q: float,
    K_max: int,   
    T_work: float = 360.0,
    lambda_w: float = 0.167,
    lambda_o: float = 0.25,
    service_time_customer: float = 10.0,
    overtime_cap: float = 60.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    S_repaired = S_destroyed_df.copy() #Starting from the destroyed solution

    # Ordered list of removed customers 
    D = removed_df.sort_values(["route_id", "position"])["customer"].tolist()

    demand_map = build_demand_map(pd.concat([S_destroyed_df, removed_df], ignore_index=True))

    while len(D) > 0: #keep inserting until all unassigned customers are assigned

        j = D[0] #taking first customer in j 
        bestCost = float("inf") #best cost is set as infinity 
        bestMove = None #best move is set as none

        current_routes = df_to_routes(S_repaired) #current routes of S_repaired 


        
        # Try inserting into best possible routes and positions
        for rid, seq in current_routes.items(): #Try all existing routes 
            for pos in range(len(seq) + 1): #Loop over all insertion positions

                new_routes = {r: s.copy() for r, s in current_routes.items()} #creating a candidate solution
                new_routes[rid].insert(pos, j)

                # Capacity check (only modified route)
                if sum(demand_map.get(c, 0.0) for c in new_routes[rid]) > Q:
                    continue

                # Time feasibility
                if not is_route_time_feasible(
                        new_routes[rid],
                        tt_by_scenario,
                        T_work,
                        overtime_cap,
                        service_time_customer
                ):
                    continue
#Compute cost increase 
                candidate_df = routes_to_df(new_routes, demand_map)
                
                old_cost = route_cost(
                    current_routes[rid],
                    tt_by_scenario,
                    probabilities,
                    T_work,
                    lambda_w,
                    lambda_o,
                    service_time_customer,
                    overtime_cap
                    )

                new_cost = route_cost(
                    new_routes[rid],
                    tt_by_scenario,
                    probabilities,
                    T_work,
                    lambda_w,
                    lambda_o,
                    service_time_customer,
                    overtime_cap
                    )
                delta = new_cost - old_cost #Marginal increase

                if delta < bestCost: #Since we are minimizing 
                    bestCost = delta
                    bestMove = (rid, pos)

        # Try inserting into a new route
        if len(current_routes) < K_max: #Only allowed if K_max does not exceed 

            new_rid = max(current_routes.keys(), default=0) + 1 #Adding an additional route
            new_route_seq = [j]

            if demand_map.get(j, 0.0) <= Q and is_route_time_feasible(
                    new_route_seq,
                    tt_by_scenario,
                    T_work,
                    overtime_cap,
                    service_time_customer
            ):

                new_routes = {r: s.copy() for r, s in current_routes.items()}
                new_routes[new_rid] = [j] #Building the new route with Customer j alone 

                
                old_cost = 0.0   # no previous route
                new_cost = route_cost([j], 
                                      tt_by_scenario,
                                      probabilities,
                                      T_work,
                                      lambda_w,
                                      lambda_o,
                                      service_time_customer,
                                      overtime_cap)

                delta = new_cost


                if delta < bestCost:
                    bestCost = delta
                    bestMove = (new_rid, 0)

        # Apply best move
        
        if bestMove is not None:

            rid, pos = bestMove
            current_routes = df_to_routes(S_repaired)
            current_routes.setdefault(rid, [])
            current_routes[rid].insert(pos, j)

            S_repaired = routes_to_df(current_routes, demand_map)
            D.pop(0)

        else:
            # No feasible insertion possible
            print(f"Customer {j} could not be inserted.")
            break

    remaining_df = pd.DataFrame({"customer": D})

    return S_repaired, remaining_df

#Regret-2 insertion operator 
def regret_2_insertion(
    S_destroyed_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    tt_by_scenario: dict[str, pd.DataFrame],
    Q: float,
    K_max: int,
    T_work: float = 360.0,
    lambda_w: float = 0.167,
    lambda_o: float = 0.25,
    service_time_customer: float = 10.0,
    overtime_cap: float = 60.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    S_repaired = S_destroyed_df.copy() #The first route for S_{repaired} is S_{destroyed}

    # Unassigned customers D
    D = removed_df.sort_values(["route_id", "position"])["customer"].tolist()

    demand_map = build_demand_map(pd.concat([S_destroyed_df, removed_df], ignore_index=True))

    while len(D) > 0: #Loop whilst D is not an empty set 

        bestRegret = -float("inf") #best regret is given minus infinity 
        chosenCustomer = None
        bestMoveForChosen = None

        current_routes = df_to_routes(S_repaired) #current routes are the routes of S_{repaired}


        # Evaluate ALL customers
        for j in D: 

            Ci = []  # list of feasible insertion costs (delta, route_id, position)

            #  Try all EXISTING routes
            for rid, seq in current_routes.items():
                for pos in range(len(seq) + 1): 

                    new_routes = {r: s.copy() for r, s in current_routes.items()}
                    new_routes[rid].insert(pos, j) #adding customer j in position pos to the current route to create the new route 

                    # Capacity check
                    if sum(demand_map.get(c, 0.0) for c in new_routes[rid]) > Q:
                        continue

                    # Time feasibility
                    if not is_route_time_feasible(
                        new_routes[rid],
                        tt_by_scenario,
                        T_work,
                        overtime_cap,
                        service_time_customer
                    ):
                        continue

                    candidate_df = routes_to_df(new_routes, demand_map)
                    old_cost = route_cost(
                        current_routes[rid],
                        tt_by_scenario,
                        probabilities,
                        T_work,
                        lambda_w,
                        lambda_o,
                        service_time_customer,
                        overtime_cap
                        )

                    new_cost = route_cost(
                        new_routes[rid],
                        tt_by_scenario,
                        probabilities,
                        T_work,
                        lambda_w,
                        lambda_o,
                        service_time_customer,
                        overtime_cap
                        )
                    delta = new_cost - old_cost
  
                
                    Ci.append((delta, rid, pos))

            # Try new route (if allowed)
            if len(current_routes) < K_max:

                new_rid = max(current_routes.keys(), default=0) + 1 #add a new route 

                if demand_map.get(j, 0.0) <= Q and is_route_time_feasible( #checking feasibility 
                    [j],
                    tt_by_scenario,
                    T_work,
                    overtime_cap,
                    service_time_customer
                ):
                    new_routes = {r: s.copy() for r, s in current_routes.items()}
                    new_routes[new_rid] = [j]


                    old_cost = 0.0   # no previous route
                    new_cost = route_cost([j], 
                                          tt_by_scenario,
                                          probabilities,
                                          T_work,
                                          lambda_w,
                                          lambda_o,
                                          service_time_customer,
                                          overtime_cap)
                    delta = new_cost

                  
                    Ci.append((delta, new_rid, 0))

            # Compute regret if feasible insertions exist
            if len(Ci) >= 1:

                Ci.sort(key=lambda x: x[0])  # sort by delta

                c1 = Ci[0][0] #c1 is the cheapest cost 

                if len(Ci) >= 2: #if a customer has more than 2 feasible positions 
                    c2 = Ci[1][0] #c2 is the second cheapest cost 
                    regret_value = c2 - c1 #calculating the regret-2 value 
                else:
                    # Only one feasible insertion has the highest priority, therefore regret value set as infinity 
                    regret_value = float("inf")

                #  Select customer with the highest regret 
                if regret_value > bestRegret:
                    bestRegret = regret_value
                    chosenCustomer = j
                    bestMoveForChosen = (Ci[0][1], Ci[0][2])  # best position

        # Apply best insertion
        if chosenCustomer is not None:

            rid, pos = bestMoveForChosen

            routes_now = df_to_routes(S_repaired)
            routes_now.setdefault(rid, [])
            routes_now[rid].insert(pos, chosenCustomer)

            S_repaired = routes_to_df(routes_now, demand_map) 
            D.remove(chosenCustomer) #remove customer from D 

        else:
            # No feasible insertion possible
            print("No feasible insertion possible.")
            break

    remaining_df = pd.DataFrame({"customer": D})

    return S_repaired, remaining_df

#Basic greedy insertion operator code 
def basic_greedy_insertion(
    S_destroyed_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    tt_by_scenario: dict[str, pd.DataFrame],
    Q: float,
    K_max: int,
    T_work: float = 360.0,
    lambda_w: float = 0.167,
    lambda_o: float = 0.25,
    service_time_customer: float = 10.0,
    overtime_cap: float = 60.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    S_repaired = S_destroyed_df.copy() #S_{repaired} gets the S_{destroyed} routes 

    # Unassigned customers list D
    D = removed_df.sort_values(["route_id", "position"])["customer"].tolist()

    # Demand map for capacity checks
    demand_map = build_demand_map(pd.concat([S_destroyed_df, removed_df], ignore_index=True))

    while len(D) > 0: #While D is not empty 

        current_routes = df_to_routes(S_repaired)

        bestCost = float("inf") #Best cost set as infinity 
        bestMove = None         # (route_id, position)
        bestCustomer = None

        # GLOBAL search over all customers and all routes/positions
        for j in D:
            
            # Try all existing routes
            for rid, seq in current_routes.items():
                for pos in range(len(seq) + 1):

                    new_routes = {r: s.copy() for r, s in current_routes.items()} #Creates candidate solutions
                    new_routes[rid].insert(pos, j)

                    # Capacity check (only modified route)
                    if sum(demand_map.get(c, 0.0) for c in new_routes[rid]) > Q:
                        continue

                    # Time feasibility
                    if not is_route_time_feasible(
                        new_routes[rid],
                        tt_by_scenario,
                        T_work,
                        overtime_cap,
                        service_time_customer
                    ):
                        continue

                    candidate_df = routes_to_df(new_routes, demand_map)
                    
                    old_cost = route_cost(
                        current_routes[rid],
                        tt_by_scenario,
                        probabilities,
                        T_work,
                        lambda_w,
                        lambda_o,
                        service_time_customer,
                        overtime_cap
                        )

                    new_cost = route_cost(
                        new_routes[rid],
                        tt_by_scenario,
                        probabilities,
                        T_work,
                        lambda_w,
                        lambda_o,
                        service_time_customer,
                        overtime_cap
                        )
                    
                    delta = new_cost - old_cost


                    if delta < bestCost:
                        bestCost = delta
                        bestMove = (rid, pos)
                        bestCustomer = j

            # Try new route (only if allowed)
            if len(current_routes) < K_max:

                new_rid = max(current_routes.keys(), default=0) + 1

                if demand_map.get(j, 0.0) <= Q and is_route_time_feasible(
                    [j],
                    tt_by_scenario,
                    T_work,
                    overtime_cap,
                    service_time_customer
                ):

                    new_routes = {r: s.copy() for r, s in current_routes.items()}
                    new_routes[new_rid] = [j]

                    
                    old_cost = 0.0   # no previous route
                    new_cost = route_cost([j], 
                                          tt_by_scenario,
                                          probabilities,
                                          T_work,
                                          lambda_w,
                                          lambda_o,
                                          service_time_customer,
                                          overtime_cap)
                    delta = new_cost


                    if delta < bestCost:
                        bestCost = delta
                        bestMove = (new_rid, 0)
                        bestCustomer = j

        
        # Apply best move
        if bestMove is None:
            print("No feasible insertion move found for remaining customers.")
            break

        rid, pos = bestMove
        routes_now = df_to_routes(S_repaired)
        routes_now.setdefault(rid, [])
        routes_now[rid].insert(pos, bestCustomer)

        S_repaired = routes_to_df(routes_now, demand_map)
        D.remove(bestCustomer) #remove the best customer j from D 

    remaining_df = pd.DataFrame({"customer": D})

    return S_repaired, remaining_df

def route_stats(
    solution_df: pd.DataFrame,
    tt_by_scenario: dict[str, pd.DataFrame],
    probabilities: dict[str, float],
    T_work: float = 360.0,
    lambda_w: float = 0.167,
    lambda_o: float = 0.25,
    service_time_customer: float = 10.0,
    overtime_cap: float = 60.0
) -> dict:

    routes = {
        int(rid): grp.sort_values("position")["customer"].tolist()
        for rid, grp in solution_df.groupby("route_id")
    }

    demand_map = build_demand_map(solution_df)

    stats = {}

    for rid, seq in routes.items():
        if not seq:
            continue

        load = sum(demand_map.get(c, 0.0) for c in seq)

        expected_time = 0.0
        expected_cost = 0.0

        # Loop over scenarios (same as objective)
        for s, tt in tt_by_scenario.items():
            p = probabilities[s]

            # travel
            travel = float(tt.loc[0, seq[0]])
            for a, b in zip(seq[:-1], seq[1:]):
                travel += float(tt.loc[a, b])
            travel += float(tt.loc[seq[-1], 0])

            service = service_time_customer * len(seq)
            T_k = travel + service

            O_k = max(0.0, T_k - T_work)
            if overtime_cap is not None:
                O_k = min(O_k, overtime_cap)

            # accumulate expected values
            expected_time += p * T_k
            expected_cost += p * (0.0456 * travel + lambda_w * T_k + lambda_o * O_k)

        stats[rid] = {
            "time": expected_time,
            "load": load,
            "cost": expected_cost   
        }

    return stats


def roulette_select(weights, rng):
    probs = np.array(weights) / np.sum(weights) #calculating the probability 
    return int(rng.choice(len(weights), p=probs))


#SA acceptance 
def SA_accept(S_current, S_candidate, cost_fn, T, rng):
    f_current = cost_fn(S_current) #f(S_current)
    f_candidate = cost_fn(S_candidate) #f(S_candidate)

    delta = f_candidate - f_current #calculating delta 

    if delta < 0:
        return S_candidate, True #always accept the solution when it is better than the current solution 
    else: #if it is not better 
        P = np.exp(-delta / T) #calculating probability of SA 
        if rng.random() < P:
            return S_candidate, True #Accept it 
        else:
            return S_current, False #reject it 

#Using roulette wheel         
def ALNS(S_initial, destroy_ops, repair_ops, cost_fn, rng,
         V=500, L=50, alpha=0.5, T=500):

    start_time = time.time()   # start timer

    S_current = S_initial.copy() #getting the routes for the current's solution
    S_best = S_initial.copy() #getting the routes for the candidate's solution (same as current initially)

    w_d = np.ones(len(destroy_ops))
    w_r = np.ones(len(repair_ops))
    
    destroy_names = ["random_removal", "worst_cost_removal", "shaw_removal"]
    repair_names = ["sequential_greedy", "regret_2", "basic_greedy"]

    pi_d = np.zeros(len(destroy_ops))
    pi_r = np.zeros(len(repair_ops))

    eps_d = np.zeros(len(destroy_ops))
    eps_r = np.zeros(len(repair_ops))

    best_cost = cost_fn(S_best) #getting the cost of the solution 
    cost_history = []
    operator_log = []

    for t in range(1, V+1):#until maximum iterations

        # Select operators (randomly)
        d_idx = roulette_select(w_d, rng)
        r_idx = roulette_select(w_r, rng)
        d_name = destroy_names[d_idx]
        r_name = repair_names[r_idx]

        destroy = destroy_ops[d_idx]
        repair = repair_ops[r_idx]

        # Destroy 
        result = destroy(S_current)
        S_minus = result.S_minus_df
        removed = result.removed_df #Set D 

        #  Repair 
        S_candidate, _ = repair(S_minus, removed)

        eps_d[d_idx] += 1 #updating number of time destroy operator has been used 
        eps_r[r_idx] += 1 #updating number of time repair operator has been used 

        
        f_current = cost_fn(S_current)   # cost of current solution
        f_candidate = cost_fn(S_candidate)   # cost of candidate solution
        f_best_before = best_cost   # best cost before this iteration is evaluated

        if f_candidate < f_best_before: # if candidate solution is better than best solution 
            phi = 6
            S_best = S_candidate.copy() #best solution is update to candidate solution 
            best_cost = f_candidate #best cost is updated to candidate's cost solution 

        elif f_candidate < f_current: #if candidate solution is not better than best solution but better than current
            phi = 3

        elif f_candidate == f_current: #if the current and candidate are equal 
            phi = 1

        else: #if candidate is less than current 
            phi = 0
            
     
            

        pi_d[d_idx] += phi
        pi_r[r_idx] += phi
        
        #Applying SA
        S_new, accepted = SA_accept(S_current, S_candidate, cost_fn, T, rng)
    
        
        operator_log.append({
            "iteration": t,
            "destroy_op": d_name,
            "repair_op": r_name,
            "accepted": accepted,
            "improved": f_candidate < f_current,
            "new_best": f_candidate < f_best_before,
            "f_current": f_current,
            "f_candidate": f_candidate,
            "f_best_before": f_best_before,
            "f_best_after": best_cost,
            "phi": phi
            })
        S_current = S_new

        # Weight update 
        if t % L == 0: #update after L iterations 
            for i in range(len(w_d)):
                if eps_d[i] > 0:
                    w_d[i] = alpha * w_d[i] + (1 - alpha) * (pi_d[i] / eps_d[i]) 
                pi_d[i] = 0
                eps_d[i] = 0

            for i in range(len(w_r)):
                if eps_r[i] > 0:
                    w_r[i] = alpha * w_r[i] + (1 - alpha) * (pi_r[i] / eps_r[i]) #equation for the weight of repair 
                pi_r[i] = 0
                eps_r[i] = 0

        # Cooling 
        T *= 0.98
        
        cost_history.append(best_cost)
        # progress print (every 10 iterations)
        if t % 10 == 0 or t == 1:
            elapsed = time.time() - start_time
            print(f"[Iter {t}/{V}] Best cost: {best_cost:.4f} | Temp: {T:.2f} | Time: {elapsed:.2f}s")

    total_time = time.time() - start_time  # end timer

    print(f"\nALNS finished in {total_time:.2f} seconds")

    return S_best, cost_history, operator_log


    
# Using the code 


if __name__ == "__main__":
    input_file = "all_initial_solutions_49customers.xlsx"
    travel_times_file = "reduced_output_49customers.xlsx"  

    tt_by_scenario, probabilities = load_travel_times_with_probabilities(travel_times_file)
    bar_t = expected_travel_time_matrix(tt_by_scenario, probabilities)
    
    # Define cost function wrapper
    cost_function = lambda sol_df: objective_cost_with_overtime(
        sol_df,
        tt_by_scenario,
        probabilities,
        T_work=360.0,
        lambda_w=0.167,
        lambda_o=0.25,
        service_time_customer=10.0,
        overtime_cap=60.0
        )


    sheet_name, S0 = choose_initial_solution_interactive(input_file)

    print(f"Using initial solution from sheet: {sheet_name}")

    best_overall_solution = None
    best_overall_cost = float("inf")
    best_overall_history = None
    best_seed = None

    all_results = []
    
    best_operator_log = None

    for seed in range(3):

        print(f"\n===== Running ALNS with seed {seed} =====")

        rng = np.random.default_rng(seed=seed)

        # redefine operators WITH this rng
        destroy_ops = [
            lambda S: random_removal_single_route(S, rng=rng),
            lambda S: worst_cost_removal_updated(S, cost_fn=cost_function, rng=rng),
            lambda S: shaw_removal_seed_based(S, bar_t=bar_t, rng=rng)
            ]

        repair_ops = [
            lambda S_minus, removed: sequential_greedy_insertion(S_minus, removed, tt_by_scenario, Q=1000, K_max=9),
            lambda S_minus, removed: regret_2_insertion(S_minus, removed, tt_by_scenario, Q=1000, K_max=9),
            lambda S_minus, removed: basic_greedy_insertion(S_minus, removed, tt_by_scenario, Q=1000, K_max=9)
            ]

        S_candidate, cost_history, operator_log = ALNS(
            S_initial=S0,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            cost_fn=cost_function,
            rng=rng,
            V=500,
            L=50,
            alpha=0.2,
            T=500
            )

        cost = cost_function(S_candidate)

        print(f"Seed {seed} final cost: {cost:.4f}")

        all_results.append((seed, cost))

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_solution = S_candidate.copy()
            best_overall_history = cost_history
            best_seed = seed
            best_operator_log = operator_log
    

    # FINAL RESULTS
    S_best = best_overall_solution
    cost_history = best_overall_history
    total_cost = best_overall_cost

    print(f"\nBest seed: {best_seed}")
    print(f"Best cost: {total_cost:.4f}")

    print("\nFinal ALNS Solution:")
    print(S_best)

    stats = route_stats(S_best, tt_by_scenario, probabilities)

    print("\nRoute statistics:")
    for rid, info in stats.items():
        print(f"Route {rid}: time={info['time']:.2f}, load={info['load']:.2f}")
        
    output_file = "ALNS_final_solution_49_V500.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Save final solution
        S_best.to_excel(writer, sheet_name="ALNS_final_solution_V500", index=False)

        # Save route statistics
        stats_df = pd.DataFrame.from_dict(stats, orient="index")
        stats_df.index.name = "route_id"
        stats_df.reset_index(inplace=True)
        stats_df.to_excel(writer, sheet_name="Route_Stats", index=False)
        results_df = pd.DataFrame(all_results, columns=["seed", "cost"])
        results_df.to_excel(writer, sheet_name="Seed_Results", index=False)
        # Convergence data
        conv_df = pd.DataFrame({
            "iteration": range(1, len(cost_history) + 1),
            "best_cost": cost_history
            })

        conv_df.to_excel(writer, sheet_name="Convergence", index=False)
        # Operator log
        op_log_df = pd.DataFrame(best_operator_log)
        op_log_df.to_excel(writer, sheet_name="Operator_Log", index=False)

    print(f"\nSolution saved to {output_file}")

    

