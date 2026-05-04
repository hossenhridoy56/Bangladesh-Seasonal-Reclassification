import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

data_path = os.path.join(root_folder, "data", "processed")
tab_path  = os.path.join(root_folder, "tables")
os.makedirs(tab_path, exist_ok=True)

df = pd.read_csv(os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv"))

features = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

labels_display = {
    'rainfall': 'Rainfall (mm)',
    'tmax': 'Tmax (°C)',
    'tmin': 'Tmin (°C)',
    'cloud': 'Cloud (okta)',
    'wind': 'Wind (m/s)',
    'humidity': 'Humidity (%)',
    'dry_bulb': 'Dry-bulb (°C)'
}

d1 = df[df['Year'] <= 1993].copy()
d2 = df[(df['Year'] >= 1994) & (df['Year'] <= 2004)].copy()
d3 = df[df['Year'] >= 2005].copy()

profile = df.groupby('Month')[features].mean().reindex(months)

scaler = StandardScaler()
X = scaler.fit_transform(profile)

km = KMeans(n_clusters=4, random_state=42, n_init=20)
km.fit(X)

centroids = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=features
)

season_map = {}

winter = (centroids['rainfall'] + centroids['tmin']).idxmin()
season_map[winter] = 'Winter'

mscore = centroids['rainfall'].rank() + centroids['humidity'].rank() + centroids['cloud'].rank()
monsoon = mscore.idxmax()
season_map[monsoon] = 'Monsoon'

rem = [i for i in centroids.index if i not in season_map]
premonsoon = centroids.loc[rem, 'tmax'].idxmax()
season_map[premonsoon] = 'Pre-Monsoon'

rem = [i for i in centroids.index if i not in season_map]
season_map[rem[0]] = 'Post-Monsoon'

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
    canonical = ['Winter','Pre-Monsoon','Monsoon','Post-Monsoon']
    mapping = {row: canonical[col] for row, col in zip(r, c)}

    return pd.Series(raw, index=months).map(mapping)

lab1 = align_decade(d1)
lab3 = align_decade(d3)

rows5 = []
for var in features:
    v1 = d1[var].dropna().values
    v2 = d2[var].dropna().values
    v3 = d3[var].dropna().values

    _, p = mannwhitneyu(v1, v3)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    rows5.append({
        'Variable': labels_display[var],
        'D1 Mean': round(v1.mean(),2),
        'D2 Mean': round(v2.mean(),2),
        'D3 Mean': round(v3.mean(),2),
        'Δ(D3-D1)': round(v3.mean()-v1.mean(),2),
        'p-value': round(p,4),
        'Sig.': sig
    })

table5 = pd.DataFrame(rows5)
table5.to_csv(os.path.join(tab_path, "Table5_Decadal_Climate_Change_Summary.csv"), index=False)

rows6 = []
for season in ['Winter','Pre-Monsoon','Monsoon','Post-Monsoon']:
    m1 = [m for m in months if lab1[m] == season]
    m3 = [m for m in months if lab3[m] == season]

    # Data-driven interpretation
    if len(m3) > len(m1):
        interp = f"{season} expanded by {len(m3)-len(m1)} month(s)"
    elif len(m3) < len(m1):
        interp = f"{season} compressed by {len(m1)-len(m3)} month(s)"
    else:
        interp = "unchanged" if set(m1) == set(m3) else f"{season} shifted"

    rows6.append({
        'Boundary Component': season + " Duration",
        'Decade 1': ', '.join(m1),
        'Decade 3': ', '.join(m3),
        'Interpretation': interp
    })

for m in ['Mar','Apr','Oct','Nov']:
    dec1 = lab1[m]
    dec3 = lab3[m]
    interp = f"{m} boundary unchanged" if dec1 == dec3 else f"{m} boundary shifted from {dec1} to {dec3}"

    rows6.append({
        'Boundary Component': m + " Boundary",
        'Decade 1': dec1,
        'Decade 3': dec3,
        'Interpretation': interp
    })

table6 = pd.DataFrame(rows6)
table6.to_csv(os.path.join(tab_path, "Table6_Seasonal_Boundary_Shift.csv"), index=False)

print("\nTABLE 6 SAVED")
print(table6.to_string(index=False))

print("\nTABLE 5 SAVED")
print(table5.to_string(index=False))

print("\nSTEP 12 FINAL COMPLETE")