import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

data_path = os.path.join(root_folder, "data", "processed")
fig_path = os.path.join(root_folder, "figures")
tab_path = os.path.join(root_folder, "tables")

for p in [fig_path, tab_path]:
    if not os.path.exists(p):
        os.makedirs(p)

input_file = os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
else:
    df = pd.read_csv(input_file)
    vars_7 = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    profile = df.groupby('Month')[vars_7].mean().reindex(month_order)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(profile)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_data)

    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=vars_7)
    rain_rank = centroids['rainfall'].rank()
    
    season_map = {}
    for i, rank in enumerate(rain_rank):
        if rank == 1:   season_map[i] = 'Winter'
        elif rank == 2: season_map[i] = 'Pre-Monsoon'
        elif rank == 3: season_map[i] = 'Post-Monsoon' 
        else:           season_map[i] = 'Monsoon'

    season_labels = [season_map[l] for l in labels]
    profile['Season_Name'] = season_labels
    profile['Cluster_ID'] = labels

    color_map = {
        'Winter': '#2166AC', 'Pre-Monsoon': '#D6604D',
        'Monsoon': '#4DAC26', 'Post-Monsoon': '#8073AC'
    }

    # Table 3 calculation
    table3 = profile.groupby('Season_Name')[vars_7].mean().reindex(['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon'])
    table3 = table3.round(3) 
    season_months = profile.reset_index().groupby('Season_Name')['Month'].apply(lambda x: ', '.join(x))
    table3['Months'] = season_months

    table3_path = os.path.join(tab_path, "Table3_Seasonal_Characteristics_FINAL.csv")
    table3.to_csv(table3_path)

    print("\n" + " TABLE 3: SEASONAL CHARACTERISTICS (3 Decimals) ".center(70, "="))
    print(table3.to_string())
    print("="*70)
    
    # FIGURE 4 SETUP
    fig4, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor='white')
    
    # মেইন টাইটেল এখানে সেট করা হয়েছে
    fig4.suptitle('Figure 4: Monthly climatological profiles of Bangladesh (1984-2014)', 
                  fontsize=16, fontweight='bold', y=0.95)

    var_titles = {'rainfall': 'Rainfall (mm)', 'tmax': 'Tmax (°C)', 'tmin': 'Tmin (°C)', 
                  'cloud': 'Cloud (oktas)', 'wind': 'Wind (m/s)', 'humidity': 'Humidity (%)', 'dry_bulb': 'Dry-Bulb (°C)'}

    for idx, (var, title) in enumerate(var_titles.items()):
        ax = axes.flatten()[idx]
        ax.bar(range(12), profile[var].values, color=[color_map[s] for s in season_labels], edgecolor='white')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(range(12))
        ax.set_xticklabels([m[0] for m in month_order])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes.flatten()[7].axis('off')
    axes.flatten()[8].axis('off')
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    axes.flatten()[8].legend(handles=patches, title='Season Clusters', loc='center')

    plt.tight_layout(rect=[0, 0, 1, 0.93]) # টাইটেলের জন্য জায়গা রাখা হয়েছে
    fig4_path = os.path.join(fig_path, 'Figure4_MonthlyProfiles.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    print(f"\n Success: Table 3 and Figure 4 saved.")

    # FIGURE 5 (HEATMAP)
    plt.figure(figsize=(14, 7))
    scaled_df = pd.DataFrame(scaled_data.T, columns=month_order, index=[v.capitalize() for v in vars_7])
    
    sns.heatmap(scaled_df, annot=True, fmt=".2f", cmap='RdBu_r', center=0, 
                linewidths=.5, cbar_kws={'label': 'Z-score'})
    
    for i, tick_label in enumerate(plt.gca().get_xticklabels()):
        tick_label.set_color(color_map[season_labels[i]])
        tick_label.set_weight('bold')

    plt.title('Standardized Monthly Climate Profiles (Z-scores, 1983–2014)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Variable', fontweight='bold')
    
    heatmap_path = os.path.join(fig_path, 'Figure5_Seasonal_Heatmap_Zscore.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {heatmap_path}")