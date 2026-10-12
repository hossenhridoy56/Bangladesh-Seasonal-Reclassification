"""
Seasonal reclassification of Bangladesh: analysis script.

Reproduces every table and figure in the manuscript. Output file names carry
the paper's numbering and go to results/ next to this script:

  results/tables_main/       Table1 ... Table8        (CSV)
  results/tables_appendix/   TableS1 ... TableS8      (CSV)
  results/figures/           Fig1 ... Fig8, FigS1     (PNG, 600 dpi)
  results/supplement/        Supp_*.csv, numbers quoted in the text that are
                             not printed as tables
  results/versions.txt       Python and package versions

Tables                                          Figures
  1   descriptive statistics                      1   analysis workflow
  2   monthly climatology                         2   study area
  3   validity indices, K = 2 to 9                3   monthly climatology
  4   cluster compositions, K = 4, 5, 6           4   correlation (a), PCA (b)
  5   seasonal characteristics (K = 4)            5   validity indices
  6   traditional seasons vs clusters             6   seasonal signatures
  7   stability and sensitivity                   7   traditional vs data-based
  8   variable importance                         8   dendrogram
  S1  per-month silhouette and stability          S1  sensitivity to K
  S2  gap statistic
  S3  stability checks, K = 3 to 6
  S4  whole-month calendar listing
  S5  station and year resampling
  S6  preprocessing variants (K = 4)
  S7  ten lowest-WCSS partitions, K = 4
  S8  calendar agreement vs contiguous null

Input: the seven raw BMD CSV files, or master_clean.csv, in SEASON_DATA_DIR
(default: the folder that holds this script).

Run:    python Complete_Analysis.py
Test:   SEASON_FAST=1 python Complete_Analysis.py    (150 dpi figures)
Extras: SEASON_VECTOR=1 also writes a PDF of each figure;
        SEASON_KEEP_OUTPUT=1 keeps files from an earlier run (they are
        deleted by default).
"""

import os
from pathlib import Path
from functools import reduce
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings, time, logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE       = Path(__file__).resolve().parent
DATA       = Path(os.environ.get("SEASON_DATA_DIR", HERE))
FAST       = bool(os.environ.get("SEASON_FAST"))
DPI        = 150 if FAST else 600
OUT_FIG    = HERE / "results" / "figures"
T_MAIN     = HERE / "results" / "tables_main"
T_APP      = HERE / "results" / "tables_appendix"
T_SUP      = HERE / "results" / "supplement"
for d in (OUT_FIG, T_MAIN, T_APP, T_SUP):
    d.mkdir(parents=True, exist_ok=True)

# Files from an earlier run are deleted so the folders only hold current output.
if not os.environ.get("SEASON_KEEP_OUTPUT"):
    for _d in (OUT_FIG, T_MAIN, T_APP, T_SUP):
        for _f in _d.iterdir():
            if _f.is_file():
                try:
                    _f.unlink()
                except PermissionError:
                    print(f"   Could not delete {_f.name}. Close it in Excel or "
                          "the image viewer, then run the script again.")
                    raise SystemExit(1)

# ----------------------------------------------------------------------
# Plot config
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#3D3D3D',
    'text.color': '#1A1A1A',
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
})

COLS = {
    'Winter': '#4472A4', 'Pre-Monsoon': '#D89A4A',
    'Monsoon': '#8E5E8E', 'Post-Monsoon': '#B87060',
    'Spring': '#5FA87A', 'Summer': '#D89A4A', 'Autumn': '#B87060',
}
COLS_SOFT = {
    'Winter': '#C9D6E3', 'Pre-Monsoon': '#F2DDB8',
    'Monsoon': '#DDC9DD', 'Post-Monsoon': '#E5C8BE',
    'Spring': '#CCE0D3', 'Summer': '#F2DDB8', 'Autumn': '#E5C8BE',
}

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
VARS6 = ['Tmax','Tmin','Rainfall','Humidity','Wind','Cloud']
PRIMARY_K = 4
SEASONS_ORDER = ['Winter','Pre-Monsoon','Monsoon','Post-Monsoon']


def plus_one_p(null, obs):
    """Plus-one permutation P: (1 + #{null >= obs}) / (1 + B). Never exactly 0."""
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


print("=" * 72)
print("  SEASON RECLASSIFICATION - MASTER ANALYSIS")
print("=" * 72)

# ======================================================================
# STEP 1 - LOAD RAW DATA
# ======================================================================
print("\n[1/8] Loading raw CSV files ...")


def load_standard(filename, var):
    df = pd.read_csv(DATA / filename, na_values=['***','****','',' ','NaN'])
    df.columns = [str(c).strip() for c in df.columns]
    rename = {'sep':'Sep','July':'Jul','June':'Jun',
              'A.Avg':'Annual','Average':'Annual',
              'Jan.':'Jan','Feb.':'Feb','Mar.':'Mar','Apr.':'Apr',
              'Aug.':'Aug','Sep.':'Sep','Oct.':'Oct','Nov.':'Nov','Dec.':'Dec'}
    df = df.rename(columns=rename)
    keep = ['Station_Raw','Year'] + MONTHS
    df = df[[c for c in keep if c in df.columns]].copy()
    for m in MONTHS:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    return df.melt(id_vars=['Station_Raw','Year'],
                   value_vars=MONTHS,
                   var_name='Month', value_name=var)


def load_drybulb(filename):
    records, year = [], None
    with open(DATA / filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('monthly'):
                for p in line.split(','):
                    p = p.strip()
                    if p.isdigit() and len(p) == 4:
                        year = int(p); break
                continue
            parts = [p.strip() for p in line.split(',')]
            if not parts or parts[0] in ('', 'Station_Raw'):
                continue
            if parts[0].startswith('Jan'):
                continue
            if parts[0].lower() in ('con.avg','con.avg.'):
                continue
            if len(parts) < 13:
                continue
            for i, m in enumerate(MONTHS):
                records.append({'Station_Raw': parts[0], 'Year': year,
                                'Month': m, 'DryBulb': parts[i+1]})
    df = pd.DataFrame(records)
    df['DryBulb'] = pd.to_numeric(df['DryBulb'], errors='coerce')
    return df


RAW_FILES = ['raw_cloud.csv','raw_rainfall.csv','raw_tmax.csv','raw_tmin.csv',
             'raw_humidity.csv','raw_wind.csv','raw_dry_bulb.csv']
if all((DATA / f).exists() for f in RAW_FILES):
    datasets = {
        'Cloud':    load_standard('raw_cloud.csv',    'Cloud'),
        'Rainfall': load_standard('raw_rainfall.csv', 'Rainfall'),
        'Tmax':     load_standard('raw_tmax.csv',     'Tmax'),
        'Tmin':     load_standard('raw_tmin.csv',     'Tmin'),
        'Humidity': load_standard('raw_humidity.csv', 'Humidity'),
        'Wind':     load_standard('raw_wind.csv',     'Wind'),
        'DryBulb':  load_drybulb('raw_dry_bulb.csv'),
    }
    for name, df in datasets.items():
        print(f"   {name:9s} -> {df.shape}")

    print("\n[2/8] Merging and cleaning ...")
    frames = [datasets[v] for v in
              ['Cloud','Rainfall','Tmax','Tmin','Humidity','Wind','DryBulb']]
    master = reduce(lambda l, r: pd.merge(l, r, on=['Station_Raw','Year','Month'],
                                           how='inner'), frames)
    master['Month'] = pd.Categorical(master['Month'], categories=MONTHS, ordered=True)
    for v in VARS6 + ['DryBulb']:
        master[v] = pd.to_numeric(master[v], errors='coerce')

    master_clean = master.dropna(subset=VARS6).copy()
    master_clean = master_clean.sort_values(['Station_Raw','Year','Month']).reset_index(drop=True)

    print(f"   Before cleaning : {master.shape}")
    print(f"   After cleaning  : {master_clean.shape}")
    print(f"   Stations        : {master_clean['Station_Raw'].nunique()}")
    print(f"   Years           : {master_clean['Year'].min()}–{master_clean['Year'].max()}")
    master_clean.to_csv(DATA / 'master_clean.csv', index=False)
    print(f"   Saved           : master_clean.csv")
else:
    print("   Raw files not found; reading master_clean.csv instead")
    master_clean = pd.read_csv(DATA / 'master_clean.csv')
    master_clean['Month'] = pd.Categorical(master_clean['Month'],
                                           categories=MONTHS, ordered=True)
    master_clean = (master_clean.dropna(subset=VARS6)
                    .sort_values(['Station_Raw','Year','Month'])
                    .reset_index(drop=True))
    print(f"   Records         : {master_clean.shape}")
    print(f"   Stations        : {master_clean['Station_Raw'].nunique()}")


# ======================================================================
# STEP 3 - TABLE 1
# ======================================================================
print("\n[3/8] Table 1: Descriptive statistics ...")
desc = master_clean[VARS6].describe().T[['mean','std','min','max']].round(2)
desc.columns = ['Mean','Std Dev','Min','Max']
desc.to_csv(T_MAIN / 'Table1_descriptive.csv')
print("\n   Table 1 - Descriptive statistics")
print(desc.to_string())


# ======================================================================
# STEP 4 - CLUSTERING
# ======================================================================
print(f"\n[4/8] National clustering (K = {PRIMARY_K}) ...")
monthly = master_clean.groupby('Month', observed=True)[VARS6].mean().reindex(MONTHS)
monthly.round(2).to_csv(T_MAIN / 'Table2_monthly_climatology.csv')

scaler = StandardScaler().fit(monthly[VARS6])
X_nat = pd.DataFrame(scaler.transform(monthly[VARS6]),
                     index=MONTHS, columns=VARS6)

import itertools

def _wcss(Z, lab):
    Z = np.asarray(Z, float)
    return float(sum(((Z[lab == c] - Z[lab == c].mean(0)) ** 2).sum() for c in np.unique(lab)))

def _contiguous_labels(k):
    out = []
    for cuts in itertools.combinations(range(12), k):
        l = np.zeros(12, int)
        for i in range(12):
            l[i] = sum(c <= i for c in cuts) % k
        out.append(l)
    return out

_Xv = X_nat.values
km_labels = min(_contiguous_labels(PRIMARY_K), key=lambda l: _wcss(_Xv, l))
km_free = KMeans(n_clusters=PRIMARY_K, random_state=42, n_init=50).fit(X_nat)
print(f"   Contiguous optimum WCSS = {_wcss(_Xv, km_labels):.4f}; "
      f"unconstrained Lloyd (seed 42) WCSS = {_wcss(_Xv, km_free.labels_):.4f}")

name_map = {}
for lab in sorted(set(km_labels)):
    months_in = set(monthly.index[km_labels == lab])
    if months_in == {'Jun','Jul','Aug','Sep'}:       name_map[lab] = 'Monsoon'
    elif months_in == {'Jan','Feb','Nov','Dec'}:     name_map[lab] = 'Winter'
    elif months_in == {'Mar','Apr','May'}:           name_map[lab] = 'Pre-Monsoon'
    elif months_in == {'Oct'}:                       name_map[lab] = 'Post-Monsoon'
    else:
        raise RuntimeError(f"Unexpected cluster composition: {sorted(months_in)}")
nat_season = {m: name_map[km_labels[i]] for i, m in enumerate(MONTHS)}


# ======================================================================
# STEP 5 - TABLES 5, 6, S4 AND SUPPLEMENT
# ======================================================================
print("\n[5/8] Tables 5 and 6: Seasonal characteristics and mapping ...")

monthly_lab = monthly.copy()
monthly_lab['Season'] = [nat_season[m] for m in MONTHS]
season_stats = (monthly_lab.groupby('Season')[VARS6]
                .mean().round(2).reindex(SEASONS_ORDER))
season_months = {s: [m for m in MONTHS if nat_season[m] == s]
                 for s in SEASONS_ORDER}
season_stats.insert(0, 'Months',
                    [' '.join(season_months[s]) for s in SEASONS_ORDER])
season_stats.to_csv(T_MAIN / 'Table5_seasonal_characteristics.csv')
print("\n   Table 5 - Seasonal characteristics")
print(season_stats.to_string())


TRAD_HALF = {
    'Grishma': [('Apr', 2), ('May', 1), ('May', 2), ('Jun', 1)],
    'Barsha':  [('Jun', 2), ('Jul', 1), ('Jul', 2), ('Aug', 1)],
    'Sharat':  [('Aug', 2), ('Sep', 1), ('Sep', 2), ('Oct', 1)],
    'Hemanta': [('Oct', 2), ('Nov', 1), ('Nov', 2), ('Dec', 1)],
    'Sheet':   [('Dec', 2), ('Jan', 1), ('Jan', 2), ('Feb', 1)],
    'Basanta': [('Feb', 2), ('Mar', 1), ('Mar', 2), ('Apr', 1)],
}
TRAD_PERIOD = {'Grishma': 'mid-Apr to mid-Jun',
               'Barsha':  'mid-Jun to mid-Aug',
               'Sharat':  'mid-Aug to mid-Oct',
               'Hemanta': 'mid-Oct to mid-Dec',
               'Sheet':   'mid-Dec to mid-Feb',
               'Basanta': 'mid-Feb to mid-Apr'}


def _agreement(halves, label_of):
    clusters = [label_of[m] for m, _ in halves]
    cnt = Counter(clusters).most_common(1)[0][1]
    parts = []
    for cl in dict.fromkeys(clusters):
        by_month = {}
        for (m, h), c in zip(halves, clusters):
            if c != cl:
                continue
            by_month.setdefault(m, []).append(h)
        labels = []
        for m, hs in by_month.items():
            hs_set = set(hs)
            if hs_set == {0} or hs_set == {1, 2}:
                labels.append(m)
            elif hs_set == {1}:
                labels.append(f"early {m}")
            elif hs_set == {2}:
                labels.append(f"late {m}")
            else:
                labels.append(m)
        parts.append(f"{cl} ({', '.join(labels)})")
    return ', '.join(parts), cnt / len(halves)


# Table 6 - mid-month mapping at K = 4
mapping_rows = []
for trad, halves in TRAD_HALF.items():
    mapped_str, agr = _agreement(halves, nat_season)
    mapping_rows.append({'Traditional Season': trad,
                         'Period (mid-month)': TRAD_PERIOD[trad],
                         'Mapped Cluster': mapped_str,
                         'Agreement': f"{100*agr:.0f}%"})
mapping = pd.DataFrame(mapping_rows)
mapping.to_csv(T_MAIN / 'Table6_traditional_mapping.csv', index=False)
print("\n   Table 6 - Traditional vs data-based mapping (mid-month, K = %d)"
      % PRIMARY_K)
print(mapping.to_string(index=False))

# Table S4 - whole-month listing (K=4) and mid-month agreement at K=3
TRAD_WHOLE = {
    'Grishma': ['Apr', 'May'],
    'Barsha':  ['Jun', 'Jul', 'Aug', 'Sep'],
    'Sharat':  ['Oct', 'Nov'],
    'Hemanta': ['Nov', 'Dec'],
    'Sheet':   ['Dec', 'Jan'],
    'Basanta': ['Feb', 'Mar'],
}
lab3 = KMeans(3, random_state=42, n_init=50).fit_predict(X_nat.values)
k3_of = {m: int(lab3[i]) for i, m in enumerate(MONTHS)}
a7_rows = []
for trad in TRAD_WHOLE:
    whole = [(m, 0) for m in TRAD_WHOLE[trad]]
    mapped_w, agr_w = _agreement(whole, nat_season)
    _, agr_3 = _agreement(TRAD_HALF[trad], k3_of)
    a7_rows.append({'Traditional Season': trad,
                    'Whole-month listing': ', '.join(TRAD_WHOLE[trad]),
                    'Cluster (K = 4)': mapped_w,
                    'Agreement, whole months, K = 4': f"{100*agr_w:.0f}%",
                    'Agreement, mid-month, K = 3':    f"{100*agr_3:.0f}%"})
tabS4 = pd.DataFrame(a7_rows)
tabS4.to_csv(T_APP / 'TableS4_whole_month_agreement.csv', index=False)
print("\n   Table S4 - whole-month listing (K = 4) and mid-month agreement at K = 3")
print(tabS4.to_string(index=False))

# Supplement - sample size per monthly profile + balanced check
tabA6 = (master_clean.groupby('Month', observed=True)
         .agg(**{'Station-months (N)': ('Year', 'size'),
                 'Stations':          ('Station_Raw', 'nunique'),
                 'Years':             ('Year', 'nunique')})
         .reindex(MONTHS).reset_index())
tabA6.to_csv(T_SUP / 'Supp_sample_size_by_month.csv', index=False)
print("\n   Supplement - station-months behind each monthly profile")
print(tabA6.to_string(index=False))

_sy = master_clean.groupby(['Station_Raw', 'Year'], observed=True).size()
n_complete = int((_sy == 12).sum())
n_incomplete = int((_sy < 12).sum())
print(f"\n   Station-years with all 12 months : {n_complete}")
print(f"   Station-years with < 12 months   : {n_incomplete}")
print(f"   Distribution of months per station-year:")
for k, v in _sy.value_counts().sort_index().items():
    print(f"        {k:>2d} months : {v} station-years")

_full = _sy[_sy == 12].reset_index()[['Station_Raw', 'Year']]
_bal = master_clean.merge(_full, on=['Station_Raw', 'Year'])
_mb = _bal.groupby('Month', observed=True)[VARS6].mean().reindex(MONTHS)
_lb = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(
    StandardScaler().fit_transform(_mb))
ari_balanced = adjusted_rand_score(km_labels, _lb)
print(f"\n   Balanced check: {n_complete} station-years with all 12 months "
      f"({len(_bal)} station-months); ARI vs primary partition = {ari_balanced:.3f}")
pd.DataFrame([{'Station_years_complete': n_complete,
               'Station_months': len(_bal),
               'ARI_vs_primary': round(ari_balanced, 3)}]).to_csv(
    T_SUP / 'Supp_balanced_check.csv', index=False)


# ======================================================================
# STEP 7 - VALIDATION
# ======================================================================
print("\n[7/8] Statistical validation ...")

sil_samples = silhouette_samples(X_nat, km_labels)
sil_table = pd.DataFrame({
    'Month': MONTHS,
    'Season': [nat_season[m] for m in MONTHS],
    'Silhouette': sil_samples.round(3),
})

rng = np.random.RandomState(42)
aris_boot = []
for b in range(1000):
    idx = rng.choice(12, size=12, replace=True)
    Xb = X_nat.values[idx]
    try:
        km_b = KMeans(n_clusters=PRIMARY_K, random_state=b, n_init=10).fit(Xb)
        aris_boot.append(adjusted_rand_score(km_labels[idx], km_b.labels_))
    except Exception:
        pass
aris_boot = np.array(aris_boot)

agg = AgglomerativeClustering(n_clusters=PRIMARY_K, linkage='ward')
agg_labels = agg.fit_predict(X_nat)
ari_hac = adjusted_rand_score(km_labels, agg_labels)

master_clean['Season'] = master_clean['Month'].map(nat_season)
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(master_clean[VARS6], master_clean['Season'])
rf_importance = pd.Series(rf.feature_importances_,
                          index=VARS6).sort_values(ascending=False)

_y = master_clean['Season'].values
_Xd = master_clean[VARS6].values
_yr = master_clean['Year'].values
perm = np.zeros(len(VARS6))
for tr, te in GroupKFold(5).split(_Xd, _y, _yr):
    _rf = RandomForestClassifier(300, random_state=42, n_jobs=-1).fit(_Xd[tr], _y[tr])
    perm += permutation_importance(_rf, _Xd[te], _y[te],
                                   n_repeats=5, random_state=0).importances_mean
perm_importance = pd.Series(perm / 5, index=VARS6).sort_values(ascending=False)
imp_df = pd.DataFrame({
    'Variable': VARS6,
    'Impurity importance': [round(rf_importance[v], 3) for v in VARS6],
    'Permutation importance (year-blocked)': [round(perm_importance[v], 3) for v in VARS6],
}).sort_values('Impurity importance', ascending=False).reset_index(drop=True)
imp_df.to_csv(T_MAIN / 'Table8_variable_importance.csv', index=False)
print("\n   Permutation importance (year-blocked):")
print(perm_importance.round(3).to_string())


# ----------------------------------------------------------------------
# STEP 7b - Revision checks
# ----------------------------------------------------------------------
print("\n[7b] Revision checks ...")
Xv = X_nat.values


def _gap(Xs, kmax=9, B=200, seed=0):
    rng_g = np.random.RandomState(seed)
    _, _, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    Xp = (Xs - Xs.mean(0)) @ Vt.T
    lo, hi = Xp.min(0), Xp.max(0)
    res = []
    for k in range(1, kmax + 1):
        ref = np.array([np.log(KMeans(k, random_state=0, n_init=20)
                               .fit(rng_g.uniform(lo, hi, Xp.shape)).inertia_)
                        for _ in range(B)])
        g = ref.mean() - np.log(KMeans(k, random_state=0, n_init=20).fit(Xp).inertia_)
        res.append((k, g, ref.std() * np.sqrt(1 + 1 / B)))
    kstar = next((res[i][0] for i in range(len(res) - 1)
                  if res[i][1] >= res[i + 1][1] - res[i + 1][2]), None)
    return res, kstar


gap_res, gap_k = _gap(Xv)
gap_map = {k: (g, se) for k, g, se in gap_res}

rows = []
for k in range(2, 10):
    l = KMeans(k, random_state=42, n_init=50).fit_predict(Xv)
    rows.append({'K': k,
                 'Silhouette':       round(silhouette_score(Xv, l), 3),
                 'Calinski_Harabasz': round(calinski_harabasz_score(Xv, l), 2),
                 'Davies_Bouldin':   round(davies_bouldin_score(Xv, l), 3),
                 'Gap':    round(gap_map[k][0], 3),
                 'Gap_SE': round(gap_map[k][1], 3)})
tabS1 = pd.DataFrame(rows)
tabS1[['K','Silhouette','Calinski_Harabasz','Davies_Bouldin']].to_csv(
    T_MAIN / 'Table3_validity_indices.csv', index=False)
tabS1[['K','Gap','Gap_SE']].to_csv(
    T_APP / 'TableS2_gap_statistic.csv', index=False)
print(f"\n   Table 3 - validation indices (Gap optimum: K = {gap_k})")
print(tabS1.to_string(index=False))


dist = lambda a_, b_: float(np.linalg.norm(Xv[MONTHS.index(a_)] - Xv[MONTHS.index(b_)]))
nov_d = {m: round(dist('Nov', m), 2) for m in ['Oct','Dec','Jan']}
feb_d = {m: round(dist('Feb', m), 2) for m in ['Jan','Mar']}
print(f"\n   Nov distances: {nov_d}   Feb distances: {feb_d}")

rng_n = np.random.RandomState(1)
null_sil = np.array([silhouette_score(
                        Xn,
                        KMeans(PRIMARY_K, random_state=0, n_init=10).fit_predict(Xn))
                     for Xn in (np.column_stack([rng_n.permutation(Xv[:, j])
                                                 for j in range(6)])
                                for _ in range(2000))])
obs_sil = silhouette_score(Xv, km_labels)
p_null = plus_one_p(null_sil, obs_sil)
print(f"\n   Silhouette (K={PRIMARY_K}) observed {obs_sil:.3f}; "
      f"shuffled-variable null mean {null_sil.mean():.3f}, "
      f"95th pct {np.percentile(null_sil, 95):.3f}, "
      f"plus-one P = {p_null:.4f}")

ym = master_clean.groupby(['Year','Month'], observed=True)[VARS6].mean().reset_index()
Z360 = StandardScaler().fit_transform(ym[VARS6])
ym['c'] = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(Z360)
modal = ym.groupby('Month', observed=True).c.agg(lambda x: x.value_counts().index[0]).reindex(MONTHS)
share = ym.groupby('Month', observed=True).c.agg(lambda x: x.value_counts(normalize=True).iloc[0]).reindex(MONTHS)
ari_360 = adjusted_rand_score(km_labels, modal.values)
stab360 = pd.DataFrame({'Month': MONTHS,
                        'Modal_cluster_share': share.round(2).values})
tabS1b = sil_table.merge(stab360, on='Month')
tabS1b.to_csv(T_APP / 'TableS1_silhouette_stability.csv', index=False)
print(f"\n   360-profile K-Means: ARI = {ari_360:.3f}")
print(stab360.T.to_string(header=False))

X5 = StandardScaler().fit_transform(monthly[[v for v in VARS6 if v != 'Wind']])
l5 = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(X5)
ari_nowind = adjusted_rand_score(km_labels, l5)
nowind = {int(c): [MONTHS[i] for i in range(12) if l5[i] == c]
          for c in sorted(set(l5))}
print(f"\n   Partition without wind: {nowind}")
print(f"   ARI = {ari_nowind:.3f}")

bd = np.array([
    [88.05,24.30],[88.15,25.10],[88.65,25.60],[89.00,26.00],
    [89.60,26.35],[90.00,26.60],[90.50,26.30],[91.30,25.20],
    [91.80,25.20],[92.10,24.90],[92.35,24.30],[92.30,23.70],
    [92.15,23.20],[92.30,22.50],[92.35,21.90],[92.20,21.30],
    [91.80,21.00],[91.40,21.10],[90.90,21.80],[90.30,21.80],
    [89.80,21.70],[89.30,22.00],[89.00,22.20],[88.70,22.50],
    [88.60,23.20],[88.20,23.70],[88.05,24.30],
])

STATION_COORDS = {
    'Ambagan (Ctg)': (22.36, 91.83), 'Barisal': (22.70, 90.37),
    'Bhola': (22.68, 90.65), 'Bogra': (24.85, 89.37),
    'Chandpur': (23.22, 90.65), 'Comilla': (23.46, 91.18),
    'Dhaka': (23.71, 90.41), 'Dinajpur': (25.63, 88.63),
    'Faridpur': (23.60, 89.85), 'Feni': (23.02, 91.40),
    'Hatia': (22.44, 91.10), 'Ishurdi': (24.13, 89.08),
    'Jessore': (23.17, 89.22), 'Khepupara': (21.90, 90.23),
    'Khulna': (22.82, 89.55), 'Kutubdia': (21.82, 91.85),
    'Madaripur': (23.17, 90.20), 'Maijdi Court': (22.87, 91.10),
    'Mongla': (22.48, 89.60), 'Mymensingh': (24.75, 90.40),
    'Patanga (Ctg)': (22.36, 91.78), 'Patuakhali': (22.35, 90.32),
    'Rajshahi': (24.37, 88.60), 'Rangamati': (22.65, 92.18),
    'Rangpur': (25.75, 89.25), 'Sandwip': (22.48, 91.48),
    'Satkhira': (22.72, 89.07), 'Sayedpur': (25.78, 88.90),
    'Sitakunda': (22.62, 91.66), 'Srimangal': (24.30, 91.73),
    'Sylhet': (24.90, 91.88), 'Tangail': (24.25, 89.92),
    'Teknaf': (20.87, 92.30),
}
stations_ordered = list(STATION_COORDS.items())


# ======================================================================
# STEP 7c - Additional tables
# ======================================================================
print("\n[7c] Additional checks ...")


def _months_of(lbl, k):
    return [[MONTHS[i] for i in range(12) if lbl[i] == c] for c in range(k)]


t2 = []
for k in (4, 5, 6):
    lbl = KMeans(k, random_state=42, n_init=50).fit_predict(Xv)
    groups = sorted(_months_of(lbl, k), key=lambda g: MONTHS.index(g[0]))
    t2.append({'k': k,
               **{f'Cluster {i+1}': ', '.join(g)
                  for i, g in enumerate(groups)}})
pd.DataFrame(t2).to_csv(T_MAIN / 'Table4_cluster_compositions.csv', index=False)

rows4 = {}
for k in (4, 5):
    lbl = KMeans(k, random_state=42, n_init=50).fit_predict(Xv)
    hac = AgglomerativeClustering(k, linkage='ward').fit_predict(Xv)
    rows4[k] = {'Silhouette':        round(silhouette_score(Xv, lbl), 3),
                'Calinski_Harabasz': round(calinski_harabasz_score(Xv, lbl), 2),
                'Davies_Bouldin':    round(davies_bouldin_score(Xv, lbl), 3),
                'ARI_KMeans_vs_HAC': round(adjusted_rand_score(lbl, hac), 3),
                'Singleton_clusters': int(sum(1 for g in _months_of(lbl, k)
                                               if len(g) == 1))}
t4 = pd.DataFrame(rows4).T
t4.index.name = 'K'
print("\n   Supplement - K = 4 vs K = 5")
print(t4.to_string())

l5a = KMeans(5, random_state=42, n_init=50).fit_predict(Xv)
nm5 = {}
for c in range(5):
    ms_ = frozenset(MONTHS[i] for i in range(12) if l5a[i] == c)
    nm5[c] = {frozenset({'Nov','Dec','Jan'}): 'Winter',
              frozenset({'Feb','Mar'}): 'Spring',
              frozenset({'Apr','May'}): 'Summer',
              frozenset({'Jun','Jul','Aug','Sep'}): 'Monsoon',
              frozenset({'Oct'}): 'Autumn'}.get(ms_, f'C{c+1}')
m5 = monthly.copy()
m5['Season'] = [nm5[l5a[i]] for i in range(12)]
t7 = m5.groupby('Season')[['Tmax','Tmin','Rainfall','Humidity']].mean().round(2)
t7.insert(0, 'Months',
          [' '.join(m for m in MONTHS if m5.loc[m, 'Season'] == s_)
           for s_ in t7.index])
t7 = t7.reindex(['Winter','Spring','Summer','Monsoon','Autumn'])
print("\n   Supplement - K = 5 characteristics")
print(t7.to_string())

rs = monthly['Rainfall']
print(f"\n   Rainfall share: Jun-Sep = "
      f"{100*rs[['Jun','Jul','Aug','Sep']].sum()/rs.sum():.1f}%, "
      f"Jun-Oct = "
      f"{100*rs[['Jun','Jul','Aug','Sep','Oct']].sum()/rs.sum():.1f}%")

_pca = PCA().fit(StandardScaler().fit_transform(master_clean[VARS6]))
_ve = _pca.explained_variance_ratio_ * 100
pca_l = pd.DataFrame(_pca.components_[:3], columns=VARS6,
                     index=[f'PC{i+1} ({_ve[i]:.1f}%)' for i in range(3)]).round(2)
pca_l.to_csv(T_SUP / 'Supp_PCA_loadings.csv')
print("\n   PCA loadings (first three components):")
print(pca_l.to_string())

abl = {}
for dv in VARS6:
    keep = [v for v in VARS6 if v != dv]
    la = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(
        StandardScaler().fit_transform(monthly[keep]))
    abl[dv] = round(adjusted_rand_score(km_labels, la), 3)
print("\n   Feature ablation:", abl)

_sub = master_clean[~master_clean['Station_Raw'].isin(['Patanga(Ctg)','Jessore'])]
_m = _sub.groupby('Month', observed=True)[VARS6].mean().reindex(MONTHS)
_l = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(
    StandardScaler().fit_transform(_m))
ari_excl = adjusted_rand_score(km_labels, _l)
_w = master_clean.groupby('Station_Raw')['Wind'].mean()
print(f"\n   Station mean wind: median {_w.median():.2f}, "
      f"min {_w.min():.2f}, max {_w.max():.2f} m/s; "
      f"Patanga {_w['Patanga(Ctg)']:.2f}, Jessore {_w['Jessore']:.2f}")
print(f"   Partition without Patanga and Jessore: ARI = {ari_excl:.3f}")


def _battery(k):
    lab_k = KMeans(k, random_state=42, n_init=50).fit(Xv).labels_
    hac_k = adjusted_rand_score(
        lab_k, AgglomerativeClustering(k, linkage='ward').fit_predict(Xv))
    r_b = np.random.RandomState(42)
    bs = []
    for b_ in range(1000):
        i_ = r_b.choice(12, 12)
        bs.append(adjusted_rand_score(
            lab_k[i_], KMeans(k, random_state=b_, n_init=10)
                        .fit(Xv[i_]).labels_))
    obs_k = silhouette_score(Xv, lab_k)
    r_n = np.random.RandomState(1)
    nl = np.array([silhouette_score(
                        Xn,
                        KMeans(k, random_state=0, n_init=10).fit_predict(Xn))
                   for Xn in (np.column_stack([r_n.permutation(Xv[:, j])
                                               for j in range(6)])
                              for _ in range(2000))])
    nw_k = adjusted_rand_score(
        lab_k, KMeans(k, random_state=42, n_init=50).fit_predict(X5))
    ym_k = ym.copy()
    ym_k['c'] = KMeans(k, random_state=42, n_init=50).fit_predict(Z360)
    md = (ym_k.groupby('Month', observed=True).c
          .agg(lambda x: x.value_counts().index[0]).reindex(MONTHS))
    return {'K': k,
            'Silhouette':        round(obs_k, 3),
            'ARI_Ward':          round(hac_k, 3),
            'Bootstrap_mean_ARI': round(float(np.mean(bs)), 3),
            'Bootstrap_2.5pct':   round(float(np.percentile(bs, 2.5)), 3),
            'Bootstrap_97.5pct':  round(float(np.percentile(bs, 97.5)), 3),
            'Null_P_plus_one':    round(plus_one_p(nl, obs_k), 4),
            'ARI_wind_dropped':   round(nw_k, 3),
            'ARI_360_profiles':   round(adjusted_rand_score(lab_k, md.values), 3)}


tabA5 = pd.DataFrame([_battery(k) for k in (3, 4, 5, 6)])
tabA5.to_csv(T_APP / 'TableS3_stability_by_K.csv', index=False)
print("\n   Table S3 - stability battery for K = 3 to 6")
print(tabA5.to_string(index=False))

t7_rows = [
    ('Ward clustering vs K-Means', f"ARI = {ari_hac:.3f}"),
    ('Bootstrap of months (1,000 samples)',
     f"Mean ARI = {aris_boot.mean():.3f}; "
     f"95% interval {np.percentile(aris_boot,2.5):.3f} to "
     f"{np.percentile(aris_boot,97.5):.3f}"),
    ('Shuffled-variable null (2,000 draws)',
     f"Observed silhouette {obs_sil:.3f}; null mean {null_sil.mean():.3f}; "
     f"plus-one P = {p_null:.4f}"),
    ('360 year-month profiles',
     f"ARI of modal assignment vs national partition = {ari_360:.3f}"),
    ('Patanga and Jessore removed', f"ARI = {ari_excl:.3f}"),
]
for dv in ['Tmax','Tmin','Rainfall','Cloud','Humidity','Wind']:
    label = 'Cloud cover' if dv == 'Cloud' else dv
    t7_rows.append((f"{label} dropped", f"ARI = {abl[dv]:.3f}"))
table7 = pd.DataFrame(t7_rows, columns=['Check','Result'])
table7.to_csv(T_MAIN / 'Table7_stability_sensitivity.csv', index=False)
print("\n   Table 7 - stability and sensitivity")
print(table7.to_string(index=False))


# ======================================================================
# STEP 7b - ROBUSTNESS CHECKS ADDED IN REVISION
# ======================================================================
print("\n[7b] Revision robustness checks ...")
import itertools

def _wcss_all(Z, labs):
    Z = np.asarray(Z, float); Z = Z - Z.mean(0)
    kmax = int(labs.max()) + 1
    tot = (Z ** 2).sum()
    out = np.empty(len(labs))
    for s0 in range(0, len(labs), 50000):
        L = labs[s0:s0 + 50000]
        acc = np.zeros(len(L))
        for c in range(kmax):
            m = (L == c).astype(float)
            n = m.sum(1)
            S = m @ Z
            acc += (S ** 2).sum(1) / np.maximum(n, 1)
        out[s0:s0 + 50000] = tot - acc
    return out

def _all_partitions(n, k):
    res = []
    def rec(i, cur, mx):
        if i == n:
            if mx + 1 == k: res.append(cur.copy())
            return
        if mx + 1 + (n - i) < k: return
        for c in range(min(mx + 2, k)):
            cur.append(c); rec(i + 1, cur, max(mx, c)); cur.pop()
    rec(0, [], -1)
    return np.array(res, dtype=np.int8)

# --- (a) S7: is the K-Means optimum unique?  Exhaustive search over ALL partitions ---
Xn = X_nat.values
_parts = _all_partitions(12, PRIMARY_K)
_w = _wcss_all(Xn, _parts)
_order = np.argsort(_w)[:10]
r1 = []
for i in _order:
    l = _parts[i].astype(int)
    grp = ' | '.join(' '.join(MONTHS[j] for j in range(12) if l[j] == c) for c in range(PRIMARY_K))
    r1.append({'Rank': len(r1) + 1, 'Partition': grp, 'WCSS': round(float(_w[i]), 4),
               'Gap_to_best_pct': round(100 * (_w[i] - _w[_order[0]]) / _w[_order[0]], 3),
               'ARI_vs_primary': round(adjusted_rand_score(km_labels, l), 3),
               'Silhouette': round(silhouette_score(Xn, l), 3) if len(set(l)) > 1 else np.nan})
tabS7 = pd.DataFrame(r1)
tabS7.to_csv(T_APP / 'TableS7_K4_exhaustive_optima.csv', index=False)
print(f"\n   S7 - {len(_parts):,} partitions into K = {PRIMARY_K} blocks searched exhaustively")
print(tabS7.to_string(index=False))

# --- (b) S8: calendar agreement vs null of random CONTIGUOUS (cyclic) partitions ---
def _mean_agree(lab12):
    lm = {m: int(lab12[i]) for i, m in enumerate(MONTHS)}
    return float(np.mean([_agreement(h, lm)[1] for h in TRAD_HALF.values()]))

_cal = np.zeros(24, int)
for _i, (_s, _h) in enumerate(TRAD_HALF.items()):
    for _m, _hh in _h: _cal[2 * MONTHS.index(_m) + (_hh - 1)] = _i

def _cyc_contig(k):
    out = []
    for cuts in itertools.combinations(range(12), k):
        l = np.zeros(12, int)
        for i in range(12):
            l[i] = sum(c <= i for c in cuts) % k
        out.append(l)
    return out

r2 = []
for k in range(2, 8):
    lk = KMeans(k, random_state=42, n_init=50).fit_predict(Xn)
    obs_a = _mean_agree(lk); obs_r = adjusted_rand_score(_cal, np.repeat(lk, 2))
    nul = _cyc_contig(k)
    na = np.array([_mean_agree(l) for l in nul]); nr = np.array([adjusted_rand_score(_cal, np.repeat(l, 2)) for l in nul])
    r2.append({'K': k, 'Obs_mean_agreement': round(obs_a, 3), 'Null_mean_agreement': round(na.mean(), 3),
               'P_agreement_ge_obs': round(float((na >= obs_a - 1e-9).mean()), 3),
               'Obs_ARI_vs_calendar': round(obs_r, 3), 'Null_mean_ARI': round(nr.mean(), 3),
               'Null_max_ARI': round(nr.max(), 3), 'P_ARI_ge_obs': round(float((nr >= obs_r - 1e-9).mean()), 3),
               'N_null_partitions': len(nul)})
tabS8 = pd.DataFrame(r2)
tabS8.to_csv(T_APP / 'TableS8_calendar_agreement_vs_null.csv', index=False)
print("\n   S8 - calendar agreement vs all contiguous cyclic partitions (half-month resolution)")
print(tabS8.to_string(index=False))

# --- (c) S5: sampling uncertainty - bootstrap STATIONS and YEARS (not months) ---
_mc = master_clean.copy(); _mc['mi'] = _mc['Month'].map({m: i for i, m in enumerate(MONTHS)}).astype(int)

def _boot_units(unit_col, B=1000, seed=42):
    units = np.sort(_mc[unit_col].unique()); uix = {u: i for i, u in enumerate(units)}
    S = np.zeros((len(units), 12, len(VARS6))); N = np.zeros((len(units), 12))
    ui = _mc[unit_col].map(uix).values; mi = _mc['mi'].values
    np.add.at(S, (ui, mi), _mc[VARS6].values); np.add.at(N, (ui, mi), 1)
    rng = np.random.RandomState(seed); res = {k: [] for k in (3, 4, 5, 6)}; parts4 = Counter()
    refs = {k: (km_labels if k == PRIMARY_K else KMeans(k, random_state=42, n_init=50).fit_predict(Xn)) for k in res}
    for b in range(B):
        w = np.bincount(rng.randint(0, len(units), len(units)), minlength=len(units)).astype(float)
        prof = np.tensordot(w, S, axes=(0, 0)) / np.maximum(w @ N, 1)[:, None]
        Zb = StandardScaler().fit_transform(prof)
        for k in res:
            lb = KMeans(k, random_state=42, n_init=10).fit_predict(Zb)
            res[k].append(adjusted_rand_score(refs[k], lb))
            if k == PRIMARY_K:
                parts4[tuple(sorted(tuple(MONTHS[i] for i in range(12) if lb[i] == c) for c in set(lb)))] += 1
    return res, parts4

r3 = []
_top = {}
for unit in ('Station_Raw', 'Year'):
    res, p4 = _boot_units(unit)
    _top[unit] = p4
    for k, v in res.items():
        v = np.array(v)
        r3.append({'Resampled_unit': 'stations' if unit == 'Station_Raw' else 'years', 'K': k,
                   'Mean_ARI': round(v.mean(), 3), 'P2.5': round(np.percentile(v, 2.5), 3),
                   'P97.5': round(np.percentile(v, 97.5), 3), 'Share_identical_to_reference': round(float((v > 0.9999).mean()), 3)})
tabS5 = pd.DataFrame(r3)
tabS5.to_csv(T_APP / 'TableS5_bootstrap_stations_years.csv', index=False)
print("\n   S5 - bootstrap of stations (33) and years (30); month profiles recomputed each time")
print(tabS5.to_string(index=False))
_rows = []
for unit, p4 in _top.items():
    for part, n in p4.most_common(5):
        _rows.append({'Resampled_unit': unit, 'K4_partition': ' | '.join(' '.join(g) for g in part), 'Share': round(n / 1000, 3)})
pd.DataFrame(_rows).to_csv(T_SUP / 'Supp_K4_partition_frequencies.csv', index=False)
print(pd.DataFrame(_rows).to_string(index=False))

# --- (d) S6: preprocessing sensitivity (rainfall transform, PCA-whitened, no wind) ---
def _k4_from(prof_df, label):
    Zt = StandardScaler().fit_transform(prof_df)
    l = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(Zt)
    return {'Variant': label, 'ARI_vs_primary': round(adjusted_rand_score(km_labels, l), 3),
            'Partition': ' | '.join(' '.join(MONTHS[j] for j in range(12) if l[j] == c) for c in sorted(set(l)))}
r4 = [_k4_from(monthly[VARS6], 'Baseline (raw rainfall)')]
for nm, f in (('sqrt rainfall', np.sqrt), ('log(1+rainfall)', np.log1p)):
    d = monthly[VARS6].copy(); d['Rainfall'] = f(d['Rainfall']); r4.append(_k4_from(d, nm))
_pc = PCA(n_components=3, whiten=True).fit_transform(Xn)
_l = KMeans(PRIMARY_K, random_state=42, n_init=50).fit_predict(_pc)
r4.append({'Variant': 'First 3 PCs, whitened', 'ARI_vs_primary': round(adjusted_rand_score(km_labels, _l), 3),
           'Partition': ' | '.join(' '.join(MONTHS[j] for j in range(12) if _l[j] == c) for c in sorted(set(_l)))})
r4.append(_k4_from(monthly[[v for v in VARS6 if v != 'Wind']], 'Wind excluded'))
_rob = (master_clean.groupby('Month', observed=True)[VARS6].median().reindex(MONTHS))
r4.append(_k4_from(_rob, 'Monthly MEDIANS instead of means'))
tabS6 = pd.DataFrame(r4)
tabS6.to_csv(T_APP / 'TableS6_preprocessing_sensitivity.csv', index=False)
print("\n   S6 - preprocessing sensitivity of the K = 4 partition"); print(tabS6.to_string(index=False))

# --- (e) Supp: data quality - extremes, station wind means, missingness ---
_cols = ['Station_Raw', 'Year', 'Month']
tabR5 = pd.concat([master_clean.nlargest(10, 'Wind')[_cols + ['Wind', 'Rainfall']].assign(Flag='top wind'),
                   master_clean.nlargest(10, 'Rainfall')[_cols + ['Wind', 'Rainfall']].assign(Flag='top rainfall')])
tabR5.to_csv(T_SUP / 'Supp_extreme_records.csv', index=False)
print("\n   Supp - ten largest wind and rainfall records")
print(tabR5.to_string(index=False))
stw = master_clean.groupby('Station_Raw')['Wind'].agg(Mean='mean', Median='median', Max='max', N='size').round(2)
stw['Ratio_to_national_median'] = (stw['Mean'] / stw['Mean'].median()).round(2)
stw.sort_values('Mean', ascending=False).to_csv(T_SUP / 'Supp_station_wind.csv')
print("\n   Supp - station wind means"); print(stw.sort_values('Mean', ascending=False).head(8).to_string())
miss = (1 - master_clean.groupby('Station_Raw').size() / 360).round(3).rename('Share_missing_by_station')
miss.sort_values(ascending=False).to_csv(T_SUP / 'Supp_missingness_by_station.csv')
print("\n   Supp - missingness: worst stations"); print(miss.sort_values(ascending=False).head(5).to_string())
_dec = master_clean.assign(Decade=pd.cut(master_clean['Year'], [1984, 1994, 2004, 2014], labels=['1985-94', '1995-2004', '2005-14']))
print("   Wind by decade:", _dec.groupby('Decade', observed=True)['Wind'].mean().round(2).to_dict())
_clean = master_clean[master_clean['Wind'] <= 10]
_pm = _clean.groupby('Month', observed=True)[VARS6].mean().reindex(MONTHS)
print("   Wind > 10 m/s removed (%d records): " % (len(master_clean) - len(_clean)), _k4_from(_pm, 'wind<=10')['ARI_vs_primary'])

# --- (g) Supp: contiguity-constrained optimum vs unconstrained solutions ---
_allp = {k: _all_partitions(12, k) for k in (3, 4, 5, 6)}
r8 = []
for k in (3, 4, 5, 6):
    lc = min(_contiguous_labels(k), key=lambda l: _wcss(Xn, l))
    wk = _wcss_all(Xn, _allp[k]); gi = int(np.argmin(wk)); lg = _allp[k][gi].astype(int)
    lf = KMeans(k, random_state=42, n_init=50).fit_predict(Xn)
    fmt = lambda l: ' | '.join(' '.join(MONTHS[j] for j in range(12) if l[j] == c) for c in sorted(set(l)))
    r8.append({'K': k, 'Contiguous_optimum': fmt(lc), 'WCSS_contiguous': round(_wcss(Xn, lc), 4),
               'Global_optimum': fmt(lg), 'WCSS_global': round(float(wk[gi]), 4),
               'Lloyd_seed42': fmt(lf), 'WCSS_Lloyd': round(_wcss(Xn, lf), 4),
               'Global_is_contiguous': bool(adjusted_rand_score(lc, lg) > 0.9999)})
tabR8 = pd.DataFrame(r8)
tabR8.to_csv(T_SUP / 'Supp_contiguous_vs_global_optimum.csv', index=False)
print("\n   Supp - contiguity-constrained vs unconstrained optimum"); print(tabR8.to_string(index=False))

# --- (h) Supp: wind record by station and decade ---
_wd = master_clean.assign(Decade=pd.cut(master_clean['Year'], [1984, 1994, 2004, 2014], labels=['1985-94', '1995-2004', '2005-14']))
tabR9 = _wd.groupby(['Station_Raw', 'Decade'], observed=True)['Wind'].mean().unstack().round(2)
tabR9['Change_pct'] = (100 * (tabR9['2005-14'] / tabR9['1985-94'] - 1)).round(0)
tabR9.sort_values('Change_pct').to_csv(T_SUP / 'Supp_wind_by_station_decade.csv')
print("\n   Supp - station wind by decade"); print(tabR9.sort_values('Change_pct').head(8).to_string())
_nat = {}
for lab_, sub in (('all stations', _wd), ('without Patanga', _wd[~_wd['Station_Raw'].str.startswith('Patanga')]),
                  ('without Patanga and Jessore', _wd[~_wd['Station_Raw'].str.startswith(('Patanga', 'Jessore'))])):
    g = sub.groupby('Decade', observed=True)['Wind'].mean()
    _nat[lab_] = {'1985-94': round(g.iloc[0], 2), '2005-14': round(g.iloc[-1], 2), 'Change_pct': round(100 * (g.iloc[-1] / g.iloc[0] - 1), 1)}
print("   National wind decade means:", _nat)
_hi = master_clean[master_clean['Wind'] > 10]
print("   Records with wind > 10 m/s by station:", _hi['Station_Raw'].value_counts().to_dict())
print("   ... by year range:", _hi['Year'].min(), '-', _hi['Year'].max())


# ======================================================================
# STEP 8 - FIGURES
# ======================================================================
print("\n[8/8] Generating figures ...")


def save_all(fig, name):
    """Save one PNG per figure (600 dpi). SEASON_VECTOR=1 also writes a PDF."""
    for _ in range(3):
        try:
            fig.savefig(OUT_FIG / f"{name}.png", dpi=DPI,
                        bbox_inches='tight', facecolor='white')
            break
        except (OSError, PermissionError):
            time.sleep(0.4)
    if os.environ.get("SEASON_VECTOR"):
        fig.savefig(OUT_FIG / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   saved {name}")


# Fig 1 - workflow
def _flow_box(ax, x, y, w, h, text, fc, fs=8.2, bold=False):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.4,rounding_size=1.6',
        fc=fc, ec='#1f3b57', lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            linespacing=1.25)


def _flow_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(patches.FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>',
        mutation_scale=10, lw=1.1, color='#1f3b57'))


fig, ax = plt.subplots(figsize=(7.2, 9.6))
ax.set_xlim(0, 100); ax.set_ylim(-5, 134); ax.axis('off')
CB, CG, CO = '#EAF1F8', '#EEF6EC', '#FDF3E3'
_flow_box(ax, 22, 120, 56, 11,
          "BMD monthly station records\n33 stations, 1985 to 2014\n"
          "six variables (seven source files)", CB)
_flow_box(ax, 22, 103, 56, 11,
          f"Merge and quality control\n10,992 merged station-months\n"
          f"{len(master_clean):,} retained, no imputation", CB)
_flow_box(ax, 22, 86, 56, 11,
          "National monthly profiles\n12 months by 6 variables\n"
          "z-score standardization", CB)
_flow_box(ax, 22, 69, 56, 11,
          "Clustering of the twelve months\nK-Means (primary, contiguous),\n"
          "Ward linkage (check)", CG)
_flow_box(ax, 22, 50, 56, 13,
          "Choice of K = 2 to 9\nSilhouette, Calinski-Harabasz, "
          "Davies-Bouldin,\ngap statistic; K = 4 retained (see text)", CG)
_flow_box(ax, 22, 35, 56, 9, "Four-season partition", CG, bold=True)
for y1, y2 in [(120,114),(103,97),(86,80),(69,63),(50,44)]:
    _flow_arrow(ax, 50, y1, 50, y2)
_flow_box(ax, 81, 86, 18, 11, "PCA and\ncorrelation\nstructure", CO, fs=7.6)
_flow_arrow(ax, 78, 91.5, 81, 91.5)
for xt in (16, 50, 84):
    _flow_arrow(ax, 50, 35, xt, 27)
_flow_box(ax, 1, 6, 31, 21,
          "Stability and sensitivity\nstation and year resampling,\n"
          "shuffled-variable null,\n360 year-month profiles,\n"
          "variable ablation,\npreprocessing variants", CO, fs=7.4)
_flow_box(ax, 34.5, 12, 31, 15,
          "Variable importance\nrandom forest;\nyear-blocked permutation\n"
          "importance", CO, fs=7.4)
_flow_box(ax, 68, 12, 31, 15,
          "Comparison with the\ntraditional six-season\ncalendar", CO, fs=7.4)
for xt, y0 in ((16.5, 6), (50, 12), (83.5, 12)):
    _flow_arrow(ax, xt, y0, xt, 2.2)
_flow_box(ax, 1, -3.5, 98, 6, "Interpretation and limitations",
          CB, fs=8, bold=True)
save_all(fig, "Fig1_workflow")

# Outline: Natural Earth 1:50m admin-0 boundary for Bangladesh (public domain)
BGD_OUTLINE = [
    [[89.051,22.093],[89.056,22.186],[89.05,22.275],[88.971,22.511],[88.921,22.632],[88.927,22.671],[88.923,22.688],[88.9,22.844],[88.867,22.939],[88.851,23.041],[88.928,23.187],[88.897,23.21],[88.808,23.23],[88.724,23.255],[88.704,23.293],[88.741,23.437],[88.698,23.493],[88.636,23.55],[88.616,23.573],[88.596,23.602],[88.567,23.674],[88.623,23.826],[88.7,24.003],[88.714,24.07],[88.727,24.186],[88.734,24.231],[88.724,24.275],[88.642,24.326],[88.499,24.347],[88.397,24.389],[88.338,24.454],[88.287,24.48],[88.225,24.461],[88.146,24.486],[88.079,24.55],[88.023,24.628],[88.03,24.664],[88.045,24.713],[88.15,24.915],[88.189,24.921],[88.279,24.882],[88.313,24.882],[88.373,24.962],[88.456,25.188],[88.574,25.188],[88.678,25.18],[88.748,25.169],[88.817,25.176],[88.89,25.194],[88.93,25.223],[88.952,25.259],[88.944,25.291],[88.855,25.334],[88.82,25.366],[88.795,25.456],[88.769,25.49],[88.593,25.495],[88.502,25.537],[88.452,25.574],[88.363,25.698],[88.253,25.79],[88.147,25.811],[88.107,25.841],[88.085,25.888],[88.097,25.956],[88.129,26.018],[88.151,26.087],[88.235,26.178],[88.334,26.258],[88.378,26.312],[88.44,26.369],[88.448,26.401],[88.437,26.437],[88.386,26.472],[88.351,26.483],[88.346,26.505],[88.37,26.564],[88.418,26.572],[88.518,26.518],[88.62,26.431],[88.681,26.353],[88.683,26.292],[88.722,26.282],[88.762,26.279],[88.828,26.252],[88.896,26.26],[88.941,26.245],[88.97,26.251],[88.982,26.286],[88.948,26.338],[88.924,26.375],[88.952,26.412],[88.983,26.42],[89.019,26.41],[89.067,26.377],[89.102,26.308],[89.108,26.202],[89.186,26.106],[89.289,26.038],[89.37,26.006],[89.467,25.984],[89.55,26.005],[89.591,26.072],[89.573,26.132],[89.586,26.186],[89.619,26.216],[89.671,26.214],[89.71,26.171],[89.823,25.941],[89.8,25.84],[89.825,25.56],[89.796,25.376],[89.801,25.336],[89.814,25.305],[89.833,25.293],[89.866,25.293],[90.004,25.258],[90.12,25.22],[90.25,25.185],[90.439,25.158],[90.555,25.167],[90.613,25.168],[90.73,25.159],[91.038,25.174],[91.293,25.178],[91.397,25.152],[91.48,25.142],[91.763,25.161],[92.05,25.169],[92.205,25.111],[92.373,25.015],[92.468,24.944],[92.485,24.903],[92.475,24.869],[92.443,24.849],[92.385,24.849],[92.251,24.895],[92.228,24.881],[92.231,24.786],[92.227,24.771],[92.198,24.686],[92.117,24.494],[92.102,24.408],[92.085,24.386],[92.064,24.374],[92.001,24.371],[91.952,24.357],[91.931,24.326],[91.899,24.261],[91.877,24.195],[91.846,24.175],[91.772,24.211],[91.727,24.205],[91.669,24.19],[91.611,24.153],[91.571,24.107],[91.526,24.091],[91.393,24.1],[91.367,24.094],[91.35,24.06],[91.336,24.019],[91.232,23.92],[91.192,23.763],[91.16,23.661],[91.166,23.581],[91.254,23.374],[91.315,23.104],[91.339,23.077],[91.359,23.068],[91.369,23.075],[91.367,23.13],[91.371,23.198],[91.399,23.214],[91.436,23.2],[91.471,23.141],[91.511,23.034],[91.554,22.992],[91.62,22.98],[91.695,23.005],[91.751,23.054],[91.774,23.106],[91.758,23.21],[91.754,23.287],[91.79,23.361],[91.919,23.471],[91.938,23.505],[91.929,23.598],[91.93,23.686],[91.979,23.692],[92.044,23.678],[92.127,23.721],[92.152,23.722],[92.187,23.676],[92.246,23.684],[92.289,23.492],[92.334,23.324],[92.334,23.242],[92.341,23.07],[92.362,22.929],[92.393,22.897],[92.43,22.822],[92.464,22.734],[92.491,22.685],[92.51,22.526],[92.532,22.41],[92.561,22.048],[92.575,21.978],[92.583,21.94],[92.584,21.609],[92.593,21.467],[92.625,21.351],[92.632,21.306],[92.6,21.27],[92.569,21.263],[92.539,21.32],[92.472,21.363],[92.373,21.409],[92.331,21.44],[92.28,21.428],[92.208,21.358],[92.18,21.293],[92.192,21.202],[92.215,21.113],[92.264,21.061],[92.268,21.005],[92.286,20.932],[92.312,20.864],[92.324,20.792],[92.308,20.79],[92.248,20.884],[92.195,20.984],[92.056,21.175],[92.011,21.516],[92.008,21.685],[91.913,21.883],[91.85,22.157],[91.825,22.229],[91.858,22.317],[91.863,22.35],[91.845,22.343],[91.797,22.297],[91.734,22.407],[91.693,22.505],[91.53,22.708],[91.482,22.797],[91.48,22.885],[91.41,22.797],[91.314,22.735],[91.216,22.642],[91.151,22.614],[90.946,22.597],[90.827,22.721],[90.656,23.025],[90.634,23.094],[90.656,23.273],[90.616,23.442],[90.616,23.532],[90.604,23.591],[90.573,23.578],[90.562,23.537],[90.568,23.474],[90.556,23.422],[90.408,23.432],[90.269,23.456],[90.392,23.367],[90.523,23.346],[90.591,23.266],[90.599,23.204],[90.595,23.134],[90.528,23.085],[90.466,23.054],[90.478,22.987],[90.552,22.905],[90.462,22.882],[90.437,22.828],[90.435,22.752],[90.481,22.685],[90.498,22.635],[90.487,22.589],[90.532,22.539],[90.596,22.436],[90.616,22.362],[90.589,22.258],[90.553,22.218],[90.494,22.179],[90.356,22.048],[90.288,21.899],[90.231,21.83],[90.159,21.817],[90.131,21.847],[90.071,21.887],[90.07,21.96],[90.088,22.017],[90.21,22.157],[90.143,22.138],[90.069,22.098],[89.954,22.023],[89.918,22.116],[89.894,22.203],[89.894,22.308],[89.985,22.466],[89.882,22.388],[89.853,22.289],[89.866,22.173],[89.853,22.091],[89.812,21.983],[89.757,21.919],[89.668,21.878],[89.628,21.814],[89.569,21.767],[89.567,21.861],[89.547,21.984],[89.483,22.276],[89.469,22.213],[89.503,22.032],[89.501,21.914],[89.452,21.821],[89.354,21.721],[89.279,21.707],[89.234,21.722],[89.167,21.784],[89.094,21.873],[89.082,22.015],[89.051,22.093]],
    [[91.151,22.175],[91.045,22.105],[91.079,22.52],[91.158,22.365],[91.178,22.283],[91.151,22.175]],
    [[91.557,22.382],[91.51,22.353],[91.467,22.378],[91.411,22.476],[91.439,22.599],[91.456,22.617],[91.484,22.577],[91.523,22.491],[91.548,22.425],[91.557,22.382]],
    [[90.778,22.089],[90.604,22.054],[90.515,22.065],[90.68,22.327],[90.675,22.445],[90.649,22.541],[90.565,22.618],[90.56,22.673],[90.523,22.748],[90.503,22.835],[90.596,22.864],[90.672,22.813],[90.683,22.785],[90.699,22.714],[90.737,22.639],[90.868,22.485],[90.866,22.391],[90.83,22.16],[90.778,22.089]],
    [[91.874,21.832],[91.838,21.75],[91.82,21.81],[91.835,21.885],[91.851,21.927],[91.861,21.927],[91.883,21.884],[91.874,21.832]],
    [[91.949,21.508],[91.889,21.503],[91.859,21.533],[91.873,21.574],[91.857,21.709],[91.908,21.723],[91.934,21.722],[91.949,21.683],[91.962,21.61],[91.949,21.508]],
    [[90.642,22.963],[90.66,22.92],[90.604,22.946],[90.562,22.975],[90.536,23.015],[90.58,23.035],[90.642,22.963]]
]
# Fig 2 - study area
fig = plt.figure(figsize=(7.2, 4.4))
ax  = fig.add_axes([0.04, 0.08, 0.40, 0.86])
axL = fig.add_axes([0.50, 0.08, 0.48, 0.86])
for _poly in BGD_OUTLINE:
    _xy = np.array(_poly)
    ax.fill(_xy[:, 0], _xy[:, 1], color='#F5F6F8',
            edgecolor='#333', linewidth=0.8, zorder=1)
_nudge = {'Ambagan (Ctg)': (0.14, 0.0), 'Patanga (Ctg)': (-0.14, 0.0)}
for i, (st, (lat, lon)) in enumerate(stations_ordered, 1):
    lon = lon + _nudge.get(st, (0, 0))[0]
    lat = lat + _nudge.get(st, (0, 0))[1]
    ax.scatter(lon, lat, s=62, facecolor='#C77B58',
               edgecolor='white', linewidth=0.6, zorder=3)
    ax.text(lon, lat, str(i), fontsize=5.6, color='white',
            ha='center', va='center', zorder=4, fontweight='bold')
ax.set_xlim(87.9, 92.9); ax.set_ylim(20.4, 26.8)
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) Study area', loc='left', fontsize=10, pad=6)
ax.set_aspect('equal'); ax.tick_params(labelsize=7.5)
ax.grid(alpha=0.12, linestyle=':', linewidth=0.4)
axL.axis('off')
half = (len(stations_ordered) + 1) // 2
for col, x0 in [(stations_ordered[:half], 0.00),
                (stations_ordered[half:], 0.50)]:
    for row, (st, _) in enumerate(col):
        idx = stations_ordered.index((st, STATION_COORDS[st])) + 1
        y = 0.97 - row * 0.060
        axL.text(x0, y, f"{idx:>2d}.", fontsize=7.5,
                 ha='right', va='center', color='#555', family='monospace')
        axL.text(x0 + 0.045, y, st, fontsize=7.5,
                 ha='left', va='center', color='#222')
axL.text(0.00, 1.01, '(b) BMD stations (n = 33)',
         fontsize=10, ha='left', va='bottom')
save_all(fig, "Fig2_study_area")

# Fig 3 - climatology
tm = (master_clean.groupby('Month', observed=True)
      .agg(tmax_m=('Tmax','mean'), tmax_s=('Tmax','std'),
           tmin_m=('Tmin','mean'), tmin_s=('Tmin','std'),
           rain_m=('Rainfall','mean'), rain_s=('Rainfall','std'))
      .reindex(MONTHS))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.2),
                               gridspec_kw={'hspace': 0.42,
                                            'left':0.13, 'right':0.97,
                                            'top':0.95, 'bottom':0.09})
x = np.arange(12)
ax1.plot(x, tm['tmax_m'], 'o-', color='#C77B58',
         label='Tmax', linewidth=1.3, markersize=4.5,
         markeredgecolor='white', markeredgewidth=0.6)
ax1.fill_between(x, tm['tmax_m']-tm['tmax_s'], tm['tmax_m']+tm['tmax_s'],
                 color='#C77B58', alpha=0.15, linewidth=0)
ax1.plot(x, tm['tmin_m'], 's-', color='#4C72B0',
         label='Tmin', linewidth=1.3, markersize=4.5,
         markeredgecolor='white', markeredgewidth=0.6)
ax1.fill_between(x, tm['tmin_m']-tm['tmin_s'], tm['tmin_m']+tm['tmin_s'],
                 color='#4C72B0', alpha=0.15, linewidth=0)
ax1.set_xticks(x); ax1.set_xticklabels(MONTHS)
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('(a) Monthly temperature (mean ± 1 SD)', loc='left')
ax1.legend(loc='center right', frameon=False)
ax1.grid(alpha=0.15, linestyle=':', linewidth=0.4)
ax1.set_ylim(bottom=0)
ax2.bar(x, tm['rain_m'], color='#5FA87A',
        edgecolor='white', linewidth=0.7, width=0.65)
ax2.errorbar(x, tm['rain_m'], yerr=tm['rain_s'],
             fmt='none', ecolor='#333', capsize=2.5, linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(MONTHS)
ax2.set_ylabel('Rainfall (mm/month)')
ax2.set_title('(b) Monthly rainfall (mean ± 1 SD)', loc='left')
ax2.grid(axis='y', alpha=0.15, linestyle=':', linewidth=0.4)
ax2.set_ylim(bottom=0)
save_all(fig, "Fig3_climatology")

# Fig 4 - correlation + PCA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                               gridspec_kw={'width_ratios':[1,1],
                                            'wspace':0.42,
                                            'left':0.08, 'right':0.96,
                                            'top':0.90, 'bottom':0.14})
corr = master_clean[VARS6].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.6, linecolor='white',
            annot_kws={'fontsize':7.5},
            cbar_kws={'label':'Pearson r', 'shrink':0.72,
                      'pad':0.03, 'aspect':18}, ax=ax1)
ax1.set_title('(a) Variable correlation', loc='left')
ax1.tick_params(labelsize=7.5)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha='right')
pca = PCA().fit(StandardScaler().fit_transform(master_clean[VARS6]))
ve = pca.explained_variance_ratio_ * 100
cum = np.cumsum(ve); k = np.arange(1, len(ve)+1)
ax2.bar(k, ve, color='#4C72B0', alpha=0.85,
        edgecolor='white', linewidth=0.6, width=0.65, label='Individual')
ax2.plot(k, cum, 'o-', color='#C77B58', linewidth=1.3,
         markersize=4.5, markeredgecolor='white', markeredgewidth=0.6,
         label='Cumulative')
ax2.axhline(95, color='#999', linestyle='--', linewidth=0.6)
ax2.set_xlabel('Principal component')
ax2.set_ylabel('Variance explained (%)')
ax2.set_title('(b) PCA - variance explained', loc='left')
ax2.legend(frameon=False, loc='center right')
ax2.grid(axis='y', alpha=0.15, linestyle=':', linewidth=0.4)
ax2.set_ylim(0, 110)
save_all(fig, "Fig4_correlation_pca")

# Fig 5 - cluster validation
Ks = list(range(2, 10))
inertia, sils, chs, dbs = [], [], [], []
for k in Ks:
    m = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X_nat)
    inertia.append(m.inertia_)
    sils.append(silhouette_score(X_nat, m.labels_))
    chs.append(calinski_harabasz_score(X_nat, m.labels_))
    dbs.append(davies_bouldin_score(X_nat, m.labels_))
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0),
                         gridspec_kw={'hspace':0.45, 'wspace':0.30,
                                      'left':0.09, 'right':0.97,
                                      'top':0.94, 'bottom':0.09})
ax = axes.flat


def mark_k(a, y):
    a.axvline(PRIMARY_K, color='#AAA', linestyle='--',
              linewidth=0.7, zorder=0)
    a.text(PRIMARY_K + 0.15, y, f'K = {PRIMARY_K}',
           color='#666', fontsize=7.5, va='center')


ax[0].plot(Ks, inertia, 'o-', color='#4C72B0', linewidth=1.3,
           markersize=4.5, markeredgecolor='white', markeredgewidth=0.6)
mark_k(ax[0], np.mean(inertia))
ax[0].set_xlabel('K'); ax[0].set_ylabel('Within-cluster sum of squares')
ax[0].set_title('(a) Elbow', loc='left')
ax[0].grid(alpha=0.15, linestyle=':', linewidth=0.4)
ax[1].plot(Ks, sils, 's-', color='#5FA87A', linewidth=1.3,
           markersize=4.5, markeredgecolor='white', markeredgewidth=0.6)
mark_k(ax[1], np.max(sils) * 0.92)
ax[1].set_xlabel('K'); ax[1].set_ylabel('Silhouette')
ax[1].set_title('(b) Silhouette', loc='left')
ax[1].grid(alpha=0.15, linestyle=':', linewidth=0.4)
ax[2].plot(Ks, chs, '^-', color='#D89A4A', linewidth=1.3,
           markersize=4.5, markeredgecolor='white', markeredgewidth=0.6)
mark_k(ax[2], np.mean(chs))
ax[2].set_xlabel('K'); ax[2].set_ylabel('Calinski–Harabasz')
ax[2].set_title('(c) Calinski–Harabasz', loc='left')
ax[2].grid(alpha=0.15, linestyle=':', linewidth=0.4)
ax[3].plot(Ks, dbs, 'D-', color='#C77B58', linewidth=1.3,
           markersize=4.5, markeredgecolor='white', markeredgewidth=0.6)
mark_k(ax[3], np.mean(dbs))
ax[3].set_xlabel('K'); ax[3].set_ylabel('Davies–Bouldin')
ax[3].set_title('(d) Davies–Bouldin', loc='left')
ax[3].grid(alpha=0.15, linestyle=':', linewidth=0.4)
save_all(fig, "Fig5_validation")

# Fig 6 - signatures
sm = season_stats[VARS6]
sm_z = (sm - sm.mean()) / sm.std()
VAR_COLS = {'Tmax':'#B87060','Tmin':'#4472A4','Rainfall':'#5FA87A',
            'Humidity':'#8E5E8E','Wind':'#7A8C99','Cloud':'#B8A560'}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(SEASONS_ORDER))
for var in VARS6:
    ax.plot(x, sm_z[var], 'o-', color=VAR_COLS[var], linewidth=2.0,
            markersize=7, markeredgecolor='white', markeredgewidth=1.0,
            label=var, zorder=3)
ax.axhline(0, color='#AAA', linewidth=0.7, linestyle=':', zorder=1)
for i, s in enumerate(SEASONS_ORDER):
    ax.axvspan(i-0.4, i+0.4, color=COLS_SOFT[s], alpha=0.35, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(SEASONS_ORDER, fontsize=10, fontweight='bold')
for i, s in enumerate(SEASONS_ORDER):
    ax.get_xticklabels()[i].set_color(COLS[s])
ax.set_ylabel('Standardized value (z-score)')
ax.set_xlabel('Season')
ax.set_xlim(-0.5, len(SEASONS_ORDER) - 0.5)
ax.grid(axis='y', alpha=0.15, linestyle=':', linewidth=0.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=6, frameon=False, columnspacing=1.2, handlelength=1.5)
plt.tight_layout()
save_all(fig, "Fig6_seasonal_signatures")

# Fig 7 - traditional vs data
trad_cols = {'Grishma':'#F0B27A', 'Barsha':'#7FB3D3', 'Sharat':'#DDA07A',
             'Hemanta':'#A8D5E5', 'Sheet':'#8EC5E8', 'Basanta':'#A8D5B5'}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.2),
                               gridspec_kw={'hspace': 0.42,
                                            'left':0.10, 'right':0.98,
                                            'top':0.93, 'bottom':0.14})
for i, (name, halves_) in enumerate(TRAD_HALF.items()):
    for m, h in halves_:
        j = MONTHS.index(m) + 0.5 * (h - 1)
        ax1.barh(i, 0.5, left=j, height=0.72,
                 color=trad_cols[name], edgecolor='white', linewidth=0.4)
    ax1.text(-0.15, i, name, ha='right', va='center',
             fontsize=8, fontweight='bold', color='#333')
ax1.set_yticks([]); ax1.set_xlim(0, 12)
ax1.set_ylim(-0.5, len(TRAD_HALF)-0.5); ax1.invert_yaxis()
ax1.set_xticks(np.arange(12)+0.5); ax1.set_xticklabels(MONTHS)
ax1.set_title('(a) Traditional six seasons (mid-month boundaries)',
              loc='left', pad=6)
for s in ax1.spines.values(): s.set_visible(False)
ax1.tick_params(left=False, length=0)
for m in MONTHS:
    ax2.barh(0, 1, left=MONTHS.index(m), height=0.72,
             color=COLS[nat_season[m]], edgecolor='white', linewidth=1.0)
ax2.set_yticks([]); ax2.set_xlim(0, 12); ax2.set_ylim(-0.5, 0.5)
ax2.set_xticks(np.arange(12)+0.5); ax2.set_xticklabels(MONTHS)
ax2.set_title(f'(b) Data-based seasons (K-means, K = {PRIMARY_K})',
              loc='left', pad=6)
for s in ax2.spines.values(): s.set_visible(False)
ax2.tick_params(left=False)
handles = [patches.Patch(facecolor=COLS[s], edgecolor='white', label=s)
           for s in SEASONS_ORDER]
ax2.legend(handles=handles, loc='upper center',
           bbox_to_anchor=(0.5, -0.55), ncol=len(SEASONS_ORDER),
           frameon=False, handlelength=1.2, columnspacing=1.5,
           handletextpad=0.4)
save_all(fig, "Fig7_traditional_vs_new")

# Fig 8 - dendrogram
Z = linkage(X_nat.values, method='ward')
fig, ax = plt.subplots(figsize=(5.5, 3.6))
dendrogram(Z, labels=MONTHS, ax=ax, leaf_font_size=9,
           color_threshold=0.62*max(Z[:,2]),
           above_threshold_color='#888')
for tick in ax.get_xticklabels():
    tick.set_color(COLS[nat_season[tick.get_text()]])
    tick.set_fontweight('bold'); tick.set_fontsize(9)
ax.set_ylabel('Ward distance')
ax.grid(axis='y', alpha=0.15, linestyle=':', linewidth=0.4)
plt.tight_layout()
save_all(fig, "Fig8_dendrogram")

# Fig S1 - sensitivity to K
K_VALUES = [3, 4, 5, 6, 7]
cluster_palette = ['#4472A4','#5FA87A','#D89A4A','#8E5E8E',
                   '#B87060','#7A8C99','#B8A560']
fig, axes = plt.subplots(len(K_VALUES), 1, figsize=(7.2, 6.2),
                         gridspec_kw={'hspace': 0.75,
                                      'left': 0.09, 'right': 0.97,
                                      'top': 0.92, 'bottom': 0.08})
for ax_i, k in enumerate(K_VALUES):
    ax = axes[ax_i]
    m = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X_nat)
    lab = m.labels_
    sil = silhouette_score(X_nat, lab)
    ax.add_patch(patches.Rectangle((0, -0.5), 12, 1,
                                   facecolor='#FAFAFB',
                                   edgecolor='none', zorder=0))
    _season_of = {}
    for _r in range(PRIMARY_K):
        _mm = set(np.where(km_labels == _r)[0])
        _season_of[_r] = ('Winter' if 0 in _mm else 'Pre-Monsoon' if 3 in _mm
                          else 'Monsoon' if 6 in _mm else 'Post-Monsoon')
    _pal = [COLS[_season_of[_r]] for _r in range(PRIMARY_K)] + ['#5FA87A', '#7B8B9A', '#B8A45A']
    _ov = np.array([[int(((lab == c_) & (km_labels == r_)).sum())
                     for r_ in range(PRIMARY_K)] for c_ in range(k)])
    _rows, _cols = linear_sum_assignment(-_ov)
    _col_of = {int(r_): int(c_) for r_, c_ in zip(_rows, _cols)}
    _spare = iter(range(PRIMARY_K, len(_pal)))
    for c_ in range(k):
        if c_ not in _col_of:
            _col_of[c_] = next(_spare)
    for i in range(12):
        c = _pal[_col_of[int(lab[i])]]
        ax.barh(0, 1, left=i, height=0.72, color=c,
                edgecolor='white', linewidth=1.2, zorder=2)
    ax.text(-0.35, 0, f'K = {k}', fontsize=10, fontweight='bold',
            va='center', ha='right')
    ax.text(12.25, 0.15, 'silhouette', fontsize=7,
            va='center', ha='left', color='#888')
    ax.text(12.25, -0.18, f'{sil:.3f}', fontsize=9, fontweight='bold',
            va='center', ha='left', color=COLS['Winter'])
    sizes = [int((lab == c).sum()) for c in range(k)]
    if ax_i == 0:
        for i, m_name in enumerate(MONTHS):
            ax.text(i+0.5, 0.62, m_name, fontsize=7,
                    ha='center', va='bottom', color='#555')
        ax.set_ylim(-1.15, 0.95)
    elif ax_i == len(K_VALUES) - 1:
        for i, m_name in enumerate(MONTHS):
            ax.text(i+0.5, -0.62, m_name, fontsize=7,
                    ha='center', va='top', color='#555')
        ax.set_ylim(-1.75, 0.85)
    else:
        ax.set_ylim(-1.15, 0.85)
    ax.set_yticks([]); ax.set_xlim(-1.2, 14.2)
    ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(left=False, bottom=False)
axes[K_VALUES.index(PRIMARY_K)].add_patch(patches.Rectangle(
    (-1.2, axes[K_VALUES.index(PRIMARY_K)].get_ylim()[0]),
    15.4,
    axes[K_VALUES.index(PRIMARY_K)].get_ylim()[1]
    - axes[K_VALUES.index(PRIMARY_K)].get_ylim()[0],
    facecolor='none',
    edgecolor=COLS['Winter'],
    linewidth=1.2, linestyle='--',
    zorder=10, clip_on=False))
axes[K_VALUES.index(PRIMARY_K)].text(
    12.25, -0.55, 'primary', fontsize=7.5, va='center',
    ha='left', color=COLS['Winter'], fontweight='bold', style='italic')
save_all(fig, "FigS1_sensitivity")


# ======================================================================
# CHECK THAT EVERY TABLE AND FIGURE WAS WRITTEN
# ======================================================================
_expected = (
    [(T_MAIN, f"{n}.csv") for n in (
        "Table1_descriptive", "Table2_monthly_climatology", "Table3_validity_indices",
        "Table4_cluster_compositions", "Table5_seasonal_characteristics",
        "Table6_traditional_mapping", "Table7_stability_sensitivity",
        "Table8_variable_importance")]
    + [(T_APP, f"{n}.csv") for n in (
        "TableS1_silhouette_stability", "TableS2_gap_statistic", "TableS3_stability_by_K",
        "TableS4_whole_month_agreement", "TableS5_bootstrap_stations_years",
        "TableS6_preprocessing_sensitivity", "TableS7_K4_exhaustive_optima",
        "TableS8_calendar_agreement_vs_null")]
    + [(OUT_FIG, f"{n}.png") for n in (
        "Fig1_workflow", "Fig2_study_area", "Fig3_climatology", "Fig4_correlation_pca",
        "Fig5_validation", "Fig6_seasonal_signatures", "Fig7_traditional_vs_new",
        "Fig8_dendrogram", "FigS1_sensitivity")]
)
_missing = [n for d_, n in _expected if not (d_ / n).exists()]

# ======================================================================
# SOFTWARE VERSIONS
# ======================================================================
import sys, platform, sklearn, scipy, matplotlib as _mpl, seaborn as _sns
with open(OUT_FIG.parent / 'versions.txt', 'w') as _fh:
    _fh.write(f"python {sys.version.split()[0]} ({platform.platform()})\n")
    for _m in (np, pd, sklearn, scipy, _mpl, _sns):
        _fh.write(f"{_m.__name__} {_m.__version__}\n")

# ======================================================================
# FINAL SUMMARY
# ======================================================================
print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)
print(f"\nResults folder: {OUT_FIG.parent}")
print("   figures/            main and supplementary figures (PNG)")
print("   tables_main/        Tables 1 to 8")
print("   tables_appendix/    Tables S1 to S8")
print("   supplement/         numbers quoted in the text only")
if _missing:
    print("\nNot written (check the log above):")
    for _n in _missing:
        print(f"   {_n}")
else:
    print("\nAll 8 main tables, 8 supplementary tables and 9 figures were written.")
print("\nKey numbers:")
print(f"   Stations         : {master_clean['Station_Raw'].nunique()}")
print(f"   Records          : {len(master_clean):,}")
print(f"   Years            : {master_clean['Year'].min()} to {master_clean['Year'].max()}")
print(f"   PCA (3 comp)     : {cum[2]:.1f}% variance")
print(f"   Seasons (K={PRIMARY_K})    : " + ", ".join(SEASONS_ORDER))
print(f"   Gap optimum K    : {gap_k}")
print(f"   Null P(sil)      : {p_null:.4f}")
print(f"   Wind-free ARI    : {ari_nowind:.3f}")
print(f"   Stations removed : ARI = {ari_excl:.3f}")
print(f"   Bootstrap ARI    : {aris_boot.mean():.3f}")
print(f"   K-means vs HAC   : ARI = {ari_hac:.3f}")
print(f"   Top feature (perm, year-blocked): "
      f"{perm_importance.index[0]} ({perm_importance.iloc[0]:.3f})")
