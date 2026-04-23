import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, ".."))

data_path = os.path.join(base_path, "data", "processed")
fig_path  = os.path.join(base_path, "figures")

os.makedirs(fig_path, exist_ok=True)

df = pd.read_csv(os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv"))

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
features = ['rainfall','tmax','tmin','cloud','wind','humidity','dry_bulb']

stations = sorted(df['Station'].unique())

def get_station_season_labels(st_df):

    monthly = st_df.groupby('Month')[features].mean().reindex(months)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(monthly)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features
    )

    rain_rank = centroids['rainfall'].rank()

    season_map = {}
    for i, r in enumerate(rain_rank):
        if r == 1:
            season_map[i] = 'Winter'
        elif r == 2:
            season_map[i] = 'Pre-Monsoon'
        elif r == 3:
            season_map[i] = 'Post-Monsoon'
        else:
            season_map[i] = 'Monsoon'

    return [season_map[l] for l in labels]

# ================================
# BUILD MATRIX
# ================================
matrix = []

for st in stations:
    st_df = df[df['Station'] == st]
    matrix.append(get_station_season_labels(st_df))

matrix = pd.DataFrame(matrix, index=stations, columns=months)

# ================================
# COLOR MAP
# ================================
season_colors = {
    'Winter': '#2166AC',
    'Pre-Monsoon': '#D6604D',
    'Monsoon': '#4DAC26',
    'Post-Monsoon': '#8073AC'
}

# Convert to numeric for heatmap
season_to_num = {k:i for i,k in enumerate(season_colors.keys())}
num_matrix = matrix.replace(season_to_num)

# ================================
# PLOT
# ================================
plt.figure(figsize=(14,10))

sns.heatmap(num_matrix,
            cmap=list(season_colors.values()),
            cbar=False,
            linewidths=0.3,
            linecolor='white')

plt.title('Figure 13: Seasonal Cluster Assignments Across Bangladesh Stations',
          fontsize=14, fontweight='bold', pad=15)

plt.xlabel('Month')
plt.ylabel('Stations')

plt.xticks(rotation=0)
plt.yticks(fontsize=8)

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=color, label=season)
           for season, color in season_colors.items()]

plt.legend(handles=patches, title='Season',
           bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'Figure13_Seasonal cluster assignments of the stations of Bangladesh.png'),
            dpi=300, bbox_inches='tight')

plt.close()

print("Figure SAVED")