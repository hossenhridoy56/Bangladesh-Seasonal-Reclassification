import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

data_path = os.path.join(root_folder, "data", "processed")
fig_path = os.path.join(root_folder, "figures")
input_file = os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv")

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
else:
    df = pd.read_csv(input_file)
    features = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']
    
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5)
    plt.title('Figure 2: Multivariate Correlation Analysis', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(fig_path, 'Figure2_Correlation_Heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    pca = PCA()
    pca.fit(scaled_data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, 8), pca.explained_variance_ratio_, alpha=0.6, label='Individual Variance')
    ax.step(range(1, 8), cum_var, where='mid', color='red', linewidth=2, label='Cumulative Variance')
    
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Figure 3: PCA Feature Importance', fontsize=14, fontweight='bold')
    ax.axhline(y=0.95, color='black', linestyle='--')
    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(fig_path, 'Figure3_PCA_Variance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n--- STEP 2/4 COMPLETE ---")
    print(f"Total Variance (PC1-PC3): {cum_var[2]*100:.2f}%")
    print(f"Saved to: {fig_path}")