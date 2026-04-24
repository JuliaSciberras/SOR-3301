# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 20:07:15 2025

@author: Julia Sciberras
"""

import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#Setup

subscription_key = "Input Subscription key"  #  Replace with your Azure Maps key

locations = [
    {"name": "Imriehel", "lat": 35.895217098790866 , "lon": 14.460191964374761},
    {"name": "Iklin", "lat": 35.90920069524615 , "lon": 14.448575317131102},
    {"name": "Bidnija", "lat": 35.89469471234125 , "lon": 14.359091106257843},
    {"name": "Mellieha", "lat": 35.958257259756884, "lon": 14.35850370248314},
    {"name": "Valletta", "lat": 35.898718948022385, "lon": 14.511286491012374},
    {"name": "Rabat", "lat": 35.879166612071614,  "lon": 14.39987123012541},
    {"name": "Zejtun", "lat": 35.85695944497671,  "lon": 14.529318941365016},
    {"name": "Mtarfa", "lat": 35.89015021650688, "lon": 14.396768907886388},
    {"name": "Mosta", "lat": 35.90871812644858,  "lon": 14.421970943790319},
    {"name": "Birzebbuga", "lat": 35.82576508828384,  "lon": 14.511967489624876},
    {"name": "Kirkop", "lat": 35.84165885609389,  "lon": 14.482503423596281},
    {"name": "Siggiewi", "lat": 35.8532857498898, "lon": 14.432303199673552},
    {"name": "Qormi", "lat": 35.87706478269086, "lon": 14.45618827095117},
    {"name": "Attard", "lat": 35.89056486287803, "lon": 14.438509831333931},
    {"name": "San Gwann", "lat": 35.909691286060784, "lon": 14.45270464577748},
    {"name": "Burmarrad", "lat": 35.933479444022126, "lon": 14.413249351168451},
    {"name": "Marsaskala", "lat": 35.867596550216646, "lon": 14.554384507323084},
    {"name": "Birgu", "lat": 35.88847506842923,  "lon": 14.522243596697626},
    {"name": "Xbiex", "lat": 35.89937391060398, "lon":  14.495020516300974},
    {"name": "Ghargur", "lat": 35.923088929342036,  "lon":  14.443250865774608},
    {"name": "Bidnija", "lat": 35.92886319046876, "lon":  14.405364215373986},
    {"name": "Luqa", "lat": 35.86100619780075,  "lon":  14.490502893924706},
    {"name": "Tarxien", "lat": 35.86588012588936,  "lon":  14.508536726236336},
    {"name": "Cirkewwa", "lat": 35.97658077974988,  "lon":  14.342265977780233 },
    {"name": "Bugibba", "lat": 35.95063017612337,  "lon":  14.417537944169375 },
    {"name": "San Gwann", "lat": 35.913469282771835, "lon":  14.453104208897392},
    {"name": "Lija", "lat": 35.90186203907905,   "lon":  14.445951680134566 },
    {"name": "Birkirkara", "lat": 35.896719121915616,  "lon":  14.463280431458081  },
    {"name": "Santa Venera", "lat": 35.88775514715397,  "lon":  14.475623689969254 },
    {"name": "Qormi", "lat": 35.87692496801545, "lon":  14.474963181061486 },
    {"name": "Kirkop", "lat": 35.84144407849942, "lon": 14.482492617824683  },
    {"name": "Ghaxaq", "lat": 35.85016149241408,  "lon":  14.51389323876203 },
    {"name": "Birzebbuga", "lat": 35.82910054159968, "lon":  14.518573822492309},
    {"name": "Marsaxlokk", "lat": 35.84439181955112,  "lon":  14.53735914968842},
    {"name": "Qrendi", "lat": 35.83541250493064,  "lon":  14.457323125853222},
    {"name": "Siggiewi", "lat": 35.856365797131495,  "lon":  14.427944890513103},
    {"name": "Zebbug", "lat": 35.8698118130993,  "lon":  14.437756411089659,},
    {"name": "Rabat", "lat": 35.87791874621391,  "lon":  14.396878204598229},
    {"name": "Mtarfa", "lat": 35.890774791845885,  "lon":  14.39489209880772},
    {"name": "Mgarr", "lat": 35.92079721961905,  "lon":  14.368493171266309},
    {"name": "Swieqi", "lat": 35.92392318963868,  "lon": 14.474845956094992},
    {"name": "Sliema", "lat": 35.91112062390241,  "lon": 14.499236623771958},
    {"name": "Xghajra", "lat": 35.88897911345391,  "lon":  14.532868563155116},
    {"name": "Madliena", "lat": 35.92716892057465, "lon":  14.466034225043124},
    {"name": "Lija", "lat": 35.909101726336985,    "lon":  14.452947727259463},
    {"name": "Bahrija", "lat": 35.895077129136546, "lon":  14.345622665832627},
    {"name": "Imtarfa", "lat": 35.8915523673847,   "lon":  14.396402467868699},
    {"name": "Qormi", "lat": 35.87899819049156,    "lon":  14.464675470223467},
    {"name": "Rahal Gdid", "lat": 35.87977330299535,  "lon": 14.509367954151235},
    {"name": "Zurrieq", "lat": 35.82508832908787,  "lon":  14.479348502056425 },
    {"name": "Xemxija", "lat": 35.95129507645979,  "lon":  14.381129614336992 },
    {"name": "Manikata", "lat": 35.944012766613376,  "lon": 14.354598277552611},
    {"name": "Dingli", "lat": 5.86546885586199,  "lon":  14.374084526054197},
    {"name": "Mgarr", "lat": 35.92169978261205,  "lon":  14.365294391054316},
    {"name": "Birzebbuga", "lat": 35.81748451848853, "lon": 14.521404049439512},
    {"name": "Ghaxaq", "lat": 35.8472759352058,  "lon":  14.511995812291927 },
    {"name": "Santa Lucija", "lat": 35.86219349628153, "lon": 14.50397929023499},
    {"name": "Birgu", "lat": 35.888554018143466,  "lon": 14.521581286895223 },
    {"name": "Siggiewi", "lat": 35.858127289730426, "lon": 14.422395143241005},
    {"name": "Rabat", "lat": 35.878337927277514, "lon":  14.401094380349036  },
    {"name": "Bidnija", "lat": 35.92724615664431,  "lon": 14.400475463004101 },
    {"name": "Bahar Ic-Caghaq", "lat": 35.9363933321593, "lon": 14.44868551207815},
    {"name": "Gharghur", "lat": 35.924258027914696,  "lon":  14.434066383748494},
    {"name": "San Giljan", "lat": 35.915283298032726, "lon":  14.477884325863894},
    {"name": "Sliema", "lat": 35.9127090592057,   "lon": 14.50021103370382 },
    {"name": "Ta xbiex", "lat": 35.89953644980623, "lon": 14.495302870661266  },
    {"name": "Furjana", "lat": 35.891738799130856, "lon": 14.50371427813205},
    {"name": "Pieta", "lat": 35.89242390792172,   "lon":  14.49285280647905 },
    {"name": "Hamrun", "lat": 35.8871381311287,  "lon":  14.489289593039945 },
    {"name": "Santa Venera", "lat": 35.89013970356884, "lon":  14.476596038877126},
    {"name": "Zejtun", "lat": 35.85919909124149, "lon":  14.529070099047825},
    {"name": "Xghajra", "lat": 35.88589986430523, "lon": 14.545155506178897},
    {"name": "Mellieha", "lat": 35.96560464561018,  "lon": 14.37948055842317},
    {"name": "Gharghur", "lat": 35.924308602248914,"lon": 14.44437458978761},
    {"name": "Burmarrad", "lat": 35.93640856657418,"lon": 14.412526712882322 },
    {"name": "Madliena", "lat": 35.92723083665774, "lon": 14.465028624045653},
    {"name": "Pieta", "lat": 35.8935895292891, "lon": 14.484969970512713},
    {"name": "Siggiewi", "lat": 35.85905912377063,"lon": 14.423521158738048},
    {"name": "Qrendi", "lat": 35.83431219869021, "lon": 14.456744118045561},
    {"name": "Marsaskala", "lat": 35.86429189535874, "lon": 14.557132487219558},
    {"name": "Furjana", "lat": 35.89212772843713, "lon": 14.50359738071902},
    {"name": "Birgu", "lat": 35.88700980018479, "lon": 14.517398397270433},
    {"name": "Bormla", "lat": 35.87868213272829, "lon": 14.520122767558279},
    {"name": "Luqa", "lat": 35.86063618967011, "lon": 14.488069028725805},
    {"name": "Imtarfa", "lat": 35.89231523763538, "lon": 14.399093124223938},
    {"name": "Naxxar", "lat": 35.9175938312863, "lon": 14.439946579060763},
    {"name": "Qawra", "lat": 35.95726497470788, "lon": 14.423068547367105},
    {"name": "Xemxija", "lat": 35.951167157275776, "lon": 14.380951160310836},
    {"name": "Bahrija", "lat": 35.89550595677966,  "lon": 14.34375060938001},
    {"name": "Imtarfa", "lat": 35.89001173543889,  "lon": 14.392596433621572},
    {"name": "Gudja", "lat": 35.8502031219277, "lon": 14.500120910235971},
    {"name": "Zejtun", "lat": 35.8600818489538, "lon": 14.530412570309421 },
    {"name": "Birzebbuga", "lat": 35.81723776037815,"lon": 14.521269503426222},
    {"name": "Marsaxlokk", "lat": 35.840271582708205, "lon": 14.54127367503653},
    {"name": "Qormi", "lat": 35.87876103788819,  "lon": 14.467408674466},
    {"name": "Hamrun", "lat": 35.88710390985843, "lon": 14.484444725024051 },
    {"name": "Valletta", "lat": 35.89694927760229, "lon":  14.509551542507953},
    {"name": "Swieqi", "lat":  35.92223404643832, "lon": 14.482353518235037},
    {"name": "Attard", "lat": 35.89676062410031, "lon": 14.43704675019653},
    {"name": "Iklin", "lat": 35.91009347518728,  "lon": 14.453901450152005},
    {"name": "Bahar ic-Caghaq", "lat": 35.93686261682869, "lon": 14.462613265032376},
    {"name": "Imgarr", "lat": 35.92115264244311, "lon": 14.36768318247279},
    {"name": "Manikata", "lat": 35.94110367262477, "lon": 14.354706655258859},
    {"name": "Ghadiera", "lat": 35.97239625398476, "lon": 14.345409599942855},
    {"name": "Burmarrad", "lat": 35.92799442017693, "lon": 14.430034376727194},
    {"name": "Gudja", "lat": 35.852158354945395,  "lon": 14.489576734648754},
    {"name": "Zejtun", "lat": 35.857461027099,  "lon": 14.532572970019388},
    {"name": "Santa Lucija", "lat":35.86238209511824, "lon": 14.505811173631766},
    {"name": "Attard", "lat": 35.88860340531732, "lon": 14.439555853605253 },
    {"name": "Imtarfa", "lat": 35.88933262369388,  "lon": 14.3949040683205},
    {"name": "Qormi", "lat": 35.87897961651583, "lon": 14.468363112777105},
    {"name": "Rahal Gdid", "lat": 35.87379899385748, "lon": 14.503699875443807},
    {"name": "Kordin", "lat": 35.8804290002041,  "lon": 14.506957418292155},
    {"name": "Ghadira", "lat": 35.971110748941186, "lon": 14.34391437014468 },
    {"name": "Xghajra", "lat": 35.88871435590162, "lon": 14.547244447970202 },
    {"name": "Luqa", "lat": 35.85969349795556,  "lon": 14.490995453731168 },
    {"name": "Mqabba", "lat": 35.84459074970752,  "lon": 14.465377430812483 },
    {"name": "Gudja", "lat": 35.84779186638794,  "lon": 14.503218999046865  },
    {"name": "Dingli", "lat": 35.865241727485845,  "lon": 114.372783174652639 },
    {"name": "Mdina", "lat": 35.88524088217352,  "lon": 14.403142335144103 },
    {"name": "Naxxar", "lat": 35.91672065801362,  "lon": 14.442096130839097  },
    {"name": "Pembroke", "lat": 35.92428310159288,  "lon": 14.471108271101802 },
    {"name": "Burmarrad", "lat": 35.93577826957046, "lon": 14.40366491482623 },
    {"name": "Golden Bay", "lat": 35.936028018245665, "lon": 14.345286635617134 },
    {"name": "Cirkewwa", "lat": 35.985051266949675, "lon": 14.361938127041327},
    {"name": "Qawra", "lat": 35.955389500582406, "lon": 14.42311679236754 },
    {"name": "Ta Xbiex", "lat": 35.89984298209957, "lon":  14.49654580489728 },
    {"name": "Hamrun", "lat": 35.88520200160999,  "lon": 14.486003305780626 },
    {"name": "Isla", "lat": 35.88687093005242,  "lon":  14.517294052290136 },
    {"name": "San Pietru", "lat": 35.88617340072296,  "lon": 14.54534674945775},
    {"name": "Zonqor", "lat": 35.87071487295273, "lon": 14.565417757662518 },
    {"name": "Qrendi", "lat": 35.83841432290208, "lon": 14.454554408850715 },
    {"name": "Zebbug", "lat": 35.869395760056534,  "lon":  14.440058950963067},
    {"name": "Attard", "lat": 35.88777088443409,  "lon": 14.433908313450562 },
    {"name": "Imgarr", "lat": 35.921453690255255, "lon": 14.364960810546421 },
    {"name": "Bahrija", "lat": 35.89974564426348,  "lon": 14.336544635145936 },
    {"name": "San Giljan", "lat": 35.91517818208451,  "lon":14.48035851894447 },
    {"name": "Rahal Gdid", "lat": 35.87254139228477,  "lon": 14.503507469983123  },
    {"name": "Qormi", "lat": 35.880725186253855,  "lon": 14.467203770966552},
    {"name": "Birkirkara", "lat": 35.899591092794395,  "lon": 14.460855199882797 },
    {"name": "Mosta", "lat": 35.90693999862525,  "lon": 14.427294819736694 },
    {"name": "Buqbaqra", "lat": 35.82424021150078,"lon":  14.476154431012569 },
    {"name": "Tal-Bajjada", "lat": 35.83675135323848,  "lon": 14.424397099955168 },
    {"name": "Zejtun", "lat": 35.858907911887606,  "lon": 14.5294863086857 },
    {"name": "Kirkop", "lat": 35.84292499792382,  "lon": 14.486352281511419},
    {"name": "Cirkewwa", "lat": 35.96445634518057,  "lon": 14.378279672227885},
    {"name": "Sliema", "lat": 35.91306187225585, "lon": 14.499474992552432},
    {"name": "Tal-Ibragg", "lat": 35.916420527646245, "lon":  14.468908538618717},
    {"name": "Birzebbuga", "lat": 35.82592836469518, "lon": 14.525227334228394},
    {"name": "Marsaxlokk", "lat": 35.84185378522058,  "lon": 14.549762487575348}
   
]

CHUNK_SIZE = 25        # 25×25 = 625 cells < 700 limit
WAIT_TIME = 3          # seconds for Azure async processing
OUTPUT_FILENAME = "emter_output_filename.xlsx"


def chunk_indices(n, chunk_size):
    """Return index pairs for splitting into blocks."""
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

#The distance was generated, however this data is not used in the dissertation 
def get_distance_block(origins, destinations, departAt_str):
    """Request one distance block from Azure Maps."""
    url = (
        f"https://atlas.microsoft.com/route/matrix/json?"
        f"api-version=1.0&subscription-key={subscription_key}&departAt={departAt_str}"
    )
    body = {
        "origins": {"type": "MultiPoint", "coordinates": origins},
        "destinations": {"type": "MultiPoint", "coordinates": destinations},
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 202:
        raise RuntimeError(f"Bad response: {r.status_code}, {r.text}")

    result_url = r.headers.get("location")
    time.sleep(WAIT_TIME)
    result = requests.get(result_url).json()

    n_o, n_d = len(origins), len(destinations)
    dist_mat = np.full((n_o, n_d), np.nan)
    for i in range(n_o):
        for j in range(n_d):
            cell = result["matrix"][i][j]
            if "response" in cell and "routeSummary" in cell["response"]:
                dist_mat[i, j] = cell["response"]["routeSummary"]["lengthInMeters"] / 1000  # km
    return dist_mat


def build_full_distance_matrix(coords, departAt_str):
    """Build the complete distance matrix by combining smaller blocks."""
    n = len(coords)
    full = np.full((n, n), np.nan)
    row_chunks = chunk_indices(n, CHUNK_SIZE)
    col_chunks = chunk_indices(n, CHUNK_SIZE)
    total_blocks = len(row_chunks) * len(col_chunks)
    block_counter = 0

    for (r0, r1) in row_chunks:
        for (c0, c1) in col_chunks:
            block_counter += 1
            print(f"🔹 Block {block_counter}/{total_blocks} "
                  f"(rows {r0}:{r1}, cols {c0}:{c1})")
            block = get_distance_block(coords[r0:r1], coords[c0:c1], departAt_str)
            full[r0:r1, c0:c1] = block
    return full



#Find weekdays at any time you want, for example this was for the 7:30 am one, for the last 90 days 



end_date = datetime.utcnow()
start_date = end_date - timedelta(days=90)

dates = []
dt = start_date
while dt <= end_date:
    if dt.weekday() < 5:  # Monday–Friday only
        weekday_7_30 = dt.replace(hour=7, minute=30, second=0, microsecond=0)
        dates.append(weekday_7_30)
    dt += timedelta(days=1)

print(f" Found {len(dates)} weekday dates at time UTC.")


# Collect and average distance 


origin_coords = [[loc["lon"], loc["lat"]] for loc in locations]
location_names = [loc["name"] for loc in locations]
all_matrices = []

for d in dates:
    departAt_str = d.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n Fetching matrix for {departAt_str}")
    try:
        matrix = build_full_distance_matrix(origin_coords, departAt_str)
        all_matrices.append(matrix)
        print(f"Completed {departAt_str}")
    except Exception as e:
        print(f"Skipped {departAt_str}: {e}")

if not all_matrices:
    raise RuntimeError("No matrices were successfully collected!")

avg_matrix = np.nanmean(np.stack(all_matrices, axis=2), axis=2)

#Export to excel 

df_avg = pd.DataFrame(avg_matrix, index=location_names, columns=location_names)
df_avg.to_excel(OUTPUT_FILENAME, sheet_name="Average_Distance")

print(f"\n Average weekday distance matrix saved to '{OUTPUT_FILENAME}'")
