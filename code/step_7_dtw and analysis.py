import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, euclidean
from fastdtw import fastdtw


current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

data_path = os.path.join(root_folder, "data", "processed")
tab_path  = os.path.join(root_folder, "tables")

if not os.path.exists(tab_path):
    os.makedirs(tab_path)

input_file = os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv")

df = pd.read_csv(input_file)

features = ['rainfall','tmax','tmin','cloud','wind','humidity','dry_bulb']
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


profile = df.groupby('Month')[features].mean().reindex(months)

scaler = StandardScaler()
X = scaler.fit_transform(profile)

km = KMeans(n_clusters=4, random_state=42, n_init=20)
labels = km.fit_predict(X)

centroids = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=features
)


season_map = {}

winter = (centroids['rainfall'] + centroids['tmin']).idxmin()
season_map[winter] = 'Winter'

monsoon_score = (
    centroids['rainfall'].rank() +
    centroids['humidity'].rank() +
    centroids['cloud'].rank()
)
monsoon = monsoon_score.idxmax()
season_map[monsoon] = 'Monsoon'

remaining = [i for i in centroids.index if i not in season_map]
premonsoon = centroids.loc[remaining, 'tmax'].idxmax()
season_map[premonsoon] = 'Pre-Monsoon'

remaining = [i for i in centroids.index if i not in season_map]
season_map[remaining[0]] = 'Post-Monsoon'

canonical_order = ['Winter','Pre-Monsoon','Monsoon','Post-Monsoon']
month_season = pd.Series(labels, index=months).map(season_map)


def align_decade(dec_df):
    monthly = dec_df.groupby('Month')[features].mean().reindex(months)
    Xd = scaler.transform(monthly)
    km_d = KMeans(n_clusters=4, random_state=42, n_init=20)
    raw = km_d.fit_predict(Xd)
    dec_cent = pd.DataFrame(
        scaler.inverse_transform(km_d.cluster_centers_),
        columns=features
    )
    ref = np.vstack([centroids.loc[c].values for c in season_map])
    cost = cdist(dec_cent.values, ref)
    r, c = linear_sum_assignment(cost)
    mapping = {row:list(season_map.values())[col] for row,col in zip(r,c)}
    return pd.Series(raw,index=months).map(mapping)


d1 = df[df['Year'] <= 1993]
d2 = df[(df['Year'] >= 1994) & (df['Year'] <= 2004)]
d3 = df[df['Year'] >= 2005]

lab1 = align_decade(d1)
lab2 = align_decade(d2)
lab3 = align_decade(d3)

cluster_matrix = pd.DataFrame({
    'Decade1': lab1,
    'Decade2': lab2,
    'Decade3': lab3
}, index=months)
cluster_matrix.to_csv(os.path.join(tab_path, "Cluster_Matrix.csv"))


def get_profile(dec_df):
    return dec_df.groupby('Month')[features].mean().reindex(months)

p1 = get_profile(d1)
p2 = get_profile(d2)
p3 = get_profile(d3)

scaler_dtw = StandardScaler()
p1_s = scaler_dtw.fit_transform(p1)
p2_s = scaler_dtw.transform(p2)
p3_s = scaler_dtw.transform(p3)

def dist_func(x, y):
    return euclidean(x, y)

d12, _ = fastdtw(p1_s, p2_s, dist=dist_func)
d23, _ = fastdtw(p2_s, p3_s, dist=dist_func)
d13, _ = fastdtw(p1_s, p3_s, dist=dist_func)

d12 /= 12
d23 /= 12
d13 /= 12


dtw_table = pd.DataFrame({
    'Comparison': [
        'Decade 1 vs Decade 2',
        'Decade 2 vs Decade 3',
        'Decade 1 vs Decade 3'
    ],
    'DTW Distance': [d12, d23, d13]
}).round(6) 

dtw_table.to_csv(os.path.join(tab_path, "Table4_DTW_Results.csv"), index=False)
print("\n" + " DTW RESULTS ".center(75, "="))
print(dtw_table.to_string(index=False))
print("\n CORE ANALYSIS COMPLETE.")