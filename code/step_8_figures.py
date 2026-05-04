import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, ".."))
data_path = os.path.join(base_path, "data", "processed")
fig_path  = os.path.join(base_path, "figures")
tab_path  = os.path.join(base_path, "tables")

os.makedirs(fig_path, exist_ok=True)
os.makedirs(tab_path, exist_ok=True)

input_file = os.path.join(data_path, "FINAL_RESEARCH_READY_DATA.csv")
df = pd.read_csv(input_file)

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
vars_7 = ['rainfall','tmax','tmin','cloud','wind','humidity','dry_bulb']

d1 = df[df['Year'] <= 1993]
d2 = df[(df['Year'] >= 1994) & (df['Year'] <= 2004)]
d3 = df[df['Year'] >= 2005]

def plot_figure8():
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.linspace(0,1000,1000)

    for m, color, label in [
        ('Jul','#2166AC','July (Peak Monsoon)'),
        ('Oct','#4DAC26','October (Extended Monsoon)'),
        ('Jan','#D6604D','January (Winter)')
    ]:
        data = df[df['Month']==m]['rainfall']
        kde = gaussian_kde(data, bw_method=0.3)
        y = kde(x)

        ax.fill_between(x,y,alpha=0.4,color=color)
        ax.plot(x,y,color=color,label=label)

    ax.set_title('Figure 8: Rainfall Probability Density: Evidence for Monsoon Extension', pad=15, fontweight='bold')
    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'Figure8 Rainfall Probability Density.png'), dpi=300)
    plt.close()

def plot_figure9():
    features = ['rainfall','tmax','humidity','cloud']

    fig, axes = plt.subplots(2,2,figsize=(14,9))
    fig.suptitle('Figure 9: Decadal Changes in Climatic Variables (1983–2014)', 
                 fontsize=14, fontweight='bold', y=0.98)

    for ax, var in zip(axes.flatten(), features):
        p1 = d1.groupby('Month')[var].mean().reindex(months)
        p2 = d2.groupby('Month')[var].mean().reindex(months)
        p3 = d3.groupby('Month')[var].mean().reindex(months)

        x = range(12)
        ax.plot(x,p1,'--o',label='Decade 1')
        ax.plot(x,p2,':s',label='Decade 2')
        ax.plot(x,p3,'-^',label='Decade 3')

        ax.axvspan(4,8,alpha=0.1)
        ax.set_title(var.upper())
        ax.set_xticks(range(12))
        ax.set_xticklabels(months)
        ax.grid(alpha=0.3)

    axes[0,0].legend()
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(fig_path, 'Figure9 Decadal Changes in Climatic Variables (1983–2014).png'), dpi=300)
    plt.close()

def plot_figure10():
    variables = ['Rainfall','Max Temp','Min Temp','Cloud','Wind','Humidity','Dry Bulb']
    d12 = [1.14,1.39,1.28,1.36,2.73,1.45,0.82]
    d23 = [1.92,1.30,1.56,1.34,3.18,2.10,1.00]

    x = np.arange(len(variables))
    plt.figure(figsize=(10,5))
    plt.bar(x-0.2,d12,0.4,label='Decade 1–2')
    plt.bar(x+0.2,d23,0.4,label='Decade 2–3')

    plt.xticks(x,variables,rotation=30)
    plt.ylabel('DTW Distance')
    plt.title('Figure 10: Variable-wise DTW Distances', pad=15, fontweight='bold')
    plt.legend()
    plt.grid(axis='y',alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'Figure10 Variable-wise DTW Distances.png'), dpi=300)
    plt.close()

def plot_figure11():
    def get_cluster(df_sub):
        monthly = df_sub.groupby('Month')[vars_7].mean().reindex(months)
        scaled = StandardScaler().fit_transform(monthly)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        return km.fit_predict(scaled)

    mat = pd.DataFrame([
        get_cluster(d1),
        get_cluster(d2),
        get_cluster(d3)
    ], index=['Decade 1','Decade 2','Decade 3'], columns=months)

    # Season mapping
    season_labels = {0:'Winter', 1:'Pre-Monsoon', 2:'Monsoon', 3:'Post-Monsoon'}
    season_colors = {'Winter':'#2166AC', 'Pre-Monsoon':'#D6604D',
                     'Monsoon':'#4DAC26', 'Post-Monsoon':'#762A83'}

    # Numeric → season label
    mat_season = mat.replace(season_labels)

    # Convert to numeric colormap (map season to color index)
    color_map = mat_season.applymap(lambda s: list(season_colors.keys()).index(s))

    plt.figure(figsize=(10,5))
    sns.heatmap(color_map, annot=False, cmap=list(season_colors.values()),
                cbar=False, linewidths=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=season) 
                       for season, color in season_colors.items()]
    plt.legend(handles=legend_elements, title='Season',
               bbox_to_anchor=(1.05,1), loc='upper left')

    plt.title('Figure 11: Decadal Cluster Assignment Heatmap (Season Colors)',
              pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'Figure11_Season_Color.png'), dpi=300)
    plt.close()




def plot_figure12():
    season_map = {'Winter':0,'Pre-Monsoon':1,'Monsoon':2,'Post-Monsoon':3}
    dec1 = ['Winter','Winter','Pre-Monsoon','Pre-Monsoon','Pre-Monsoon',
            'Monsoon','Monsoon','Monsoon','Monsoon','Post-Monsoon','Post-Monsoon','Winter']
    dec3 = ['Winter','Winter','Post-Monsoon','Post-Monsoon','Pre-Monsoon',
            'Monsoon','Monsoon','Monsoon','Monsoon','Pre-Monsoon','Winter','Winter']

    y1 = [season_map[s] for s in dec1]
    y3 = [season_map[s] for s in dec3]

    x = np.arange(12)
    plt.figure(figsize=(12,5))
    plt.bar(x-0.2,y1,0.4,label='Decade 1')
    plt.bar(x+0.2,y3,0.4,label='Decade 3')

    for i in range(12):
        if y1[i] != y3[i]:
            plt.text(i,3.2,'Shift',ha='center',color='red',fontsize=9)

    plt.xticks(x,months)
    plt.yticks([0,1,2,3],['Winter','Pre-Monsoon','Monsoon','Post-Monsoon'])
    plt.ylim(0,3.5)

    plt.title('Figure 12: Seasonal Phase Shift (Decade 1 vs Decade 3)',
              pad=15, fontweight='bold')

    plt.legend()
    plt.grid(axis='y',alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'Figure12_Phase shift analysis.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_figure8()
    plot_figure9()
    plot_figure10()
    plot_figure11()
    plot_figure12()
    print(f" ALL FIGURES SAVED IN: {fig_path}")