import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

data_path = os.path.join(root_folder, "data", "processed")
fig_path = os.path.join(root_folder, "figures")
tab_path = os.path.join(root_folder, "tables")

for path in [fig_path, tab_path]:
    if not os.path.exists(path):
        os.makedirs(path)

input_path = os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv")

if not os.path.exists(input_path):
    print(f"Dataset path error: {input_path} not found.")
else:
    df = pd.read_csv(input_path)
    features = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    profile = df.groupby('Month')[features].mean().reindex(month_order)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profile)

    # ======================================================================
    # FIGURE 6: CLUSTER VALIDATION METRICS 
    # ======================================================================
    ks, sils, dbs, chs = [], [], [], []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        ks.append(k)
        sils.append(silhouette_score(scaled, labels))
        dbs.append(davies_bouldin_score(scaled, labels))
        chs.append(calinski_harabasz_score(scaled, labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    fig.suptitle('Figure 6: Cluster Validation Metrics for Optimal Season Selection', fontsize=14, fontweight='bold')

    configs = [
        (axes[0], sils, 'Silhouette Score\n(Higher is better)', '#2166AC'),
        (axes[1], dbs,  'Davies-Bouldin Index\n(Lower is better)', '#D6604D'),
        (axes[2], chs,  'Calinski-Harabasz Index\n(Higher is better)', '#2166AC'),
    ]

    for i, (ax, vals, title, color) in enumerate(configs):
        ax.plot(ks, vals, 'o-', color=color, linewidth=2, markersize=8, zorder=3)
        val_k4 = vals[2] # Value for k=4
        
        ax.axvline(x=4, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.scatter([4], [val_k4], color='red', s=100, zorder=5)
        ax.annotate(f'k=4\n({val_k4:.3f})', xy=(4, val_k4), xytext=(4.2, val_k4),
                    fontsize=9, color='red', fontweight='bold', 
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_xticks(ks)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_path, 'Figure6_Validation_Metrics.png'), dpi=300)
    plt.close()

    # ======================================================================
    # FIGURE 7: DENDROGRAM 
    # ======================================================================
    Z = sch.linkage(scaled, method='ward')
    stable_dist = 2.53 

    plt.figure(figsize=(10, 7), facecolor='white')
    dendro = sch.dendrogram(
        Z, 
        labels=month_order, 
        leaf_rotation=0, 
        leaf_font_size=11,
        color_threshold=2.55 
    )

    plt.axhline(y=stable_dist, color='red', linestyle='--', 
                label=f'Stable Cluster Formation (Distance = {stable_dist})')
    
    plt.title('Figure 7: Hierarchical Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Euclidean Linkage Distance (Standardized)', fontsize=12)
    plt.xlabel('Months')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    plt.savefig(os.path.join(fig_path, 'Figure7_Dendrogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ======================================================================
    # TABLE 2: K-MEANS VS GMM COMPARISON
    # ======================================================================
    gmm = GaussianMixture(n_components=4, random_state=42).fit(scaled)
    gmm_lab = gmm.predict(scaled)

    table2_df = pd.DataFrame({
        'Algorithm': ['K-Means', 'GMM'],
        'Silhouette Score ↑': [sils[2], silhouette_score(scaled, gmm_lab)],
        'Davies–Bouldin Index ↓': [dbs[2], davies_bouldin_score(scaled, gmm_lab)],
        'Calinski–Harabasz Index ↑': [chs[2], calinski_harabasz_score(scaled, gmm_lab)]
    })
    
    table2_df = table2_df.round(3)
    table2_df.to_csv(os.path.join(tab_path, "Table2_Algorithm_Comparison.csv"), index=False)
    
    print("\n--- STEP 6: VALIDATION COMPLETE ---")
    print(f"Figure 6 & 7 saved in: {fig_path}")
    print(f"Table 2 saved in: {tab_path}")
    print("\n" + "="*75)
    print("FINAL ALGORITHM COMPARISON (k=4, 3 Decimals)")
    print("-" * 75)
    print(table2_df.to_string(index=False))
    print("="*75)