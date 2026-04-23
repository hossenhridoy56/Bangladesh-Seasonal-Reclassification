import pandas as pd
import numpy as np
import os
import re

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

output_dir = os.path.join(root_folder, "data", "processed")
output_step1 = os.path.join(output_dir, "master_step1_raw_integrated.csv")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def parse_dry_bulb(filename):
    data = []
    if not os.path.exists(filename): return pd.DataFrame()
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    curr_yr = None
    for line in lines:
        match = re.search(r'of,(\d{4})', line.strip())
        if match: curr_yr = match.group(1); continue
        row = line.strip().split(',')
        if curr_yr and len(row) >= 13 and row[0] not in ['Station_Raw', 'Monthly', 'Average']:
            entry = {'Station': row[0].strip(), 'Year': curr_yr}
            for idx, m in enumerate(month_order):
                val = str(row[idx+1]).replace('*', '').strip()
                try: entry[m] = float(val) if val else np.nan
                except: entry[m] = np.nan
            data.append(entry)
    return pd.DataFrame(data)

def prepare_for_merge(df, var_name):
    df.columns = [str(c).strip().replace('.', '') for c in df.columns]
    rename_dict = {}
    for col in df.columns:
        c_low = col.lower()
        if 'station' in c_low: rename_dict[col] = 'Station'
        elif 'year' in c_low: rename_dict[col] = 'Year'
        for m in month_order:
            if c_low.startswith(m.lower()): rename_dict[col] = m

    df = df.rename(columns=rename_dict)
    if 'Station' not in df.columns or 'Year' not in df.columns: return pd.DataFrame()

    present_months = [m for m in month_order if m in df.columns]
    if len(present_months) == 0: return pd.DataFrame()

    df_long = df.melt(id_vars=['Station', 'Year'], 
                      value_vars=present_months, 
                      var_name='Month', 
                      value_name=var_name)
    
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long['Station'] = df_long['Station'].astype(str).str.strip().str.title()
    df_long[var_name] = pd.to_numeric(df_long[var_name].astype(str).str.replace('*', '', regex=False).str.strip(), errors='coerce')
    
    return df_long.dropna(subset=['Station', 'Year'])

print(f"--- Step 1: Integrating Variables from {root_folder} ---")

raw_data_path = os.path.join(root_folder, "data", "raw")
search_path = raw_data_path if os.path.exists(raw_data_path) else root_folder
all_files = os.listdir(search_path)

file_map = {
    "rainfall": ["rain", "precip"],
    "tmax": ["tmax", "maximum", "max_temp"],
    "tmin": ["tmin", "minimum", "min_temp"],
    "cloud": ["cloud"],
    "wind": ["wind"],
    "humidity": ["humid"]
}

master_df = None

for var_key, keywords in file_map.items():
    target_file = None
    for f in all_files:
        if any(kw in f.lower() for kw in keywords) and (f.endswith('.csv') or f.endswith('.xlsx')):
            target_file = f
            break
    
    if target_file:
        full_path = os.path.join(search_path, target_file)
        try:
            df_raw = pd.read_excel(full_path) if target_file.endswith('.xlsx') else pd.read_csv(full_path, encoding_errors='ignore')
            df_clean = prepare_for_merge(df_raw, var_key)
            if not df_clean.empty:
                if master_df is None: master_df = df_clean
                else: master_df = pd.merge(master_df, df_clean, on=['Station', 'Year', 'Month'], how='outer')
                print(f"Integrated: {var_key}")
        except Exception as e:
            print(f"Error processing {target_file}: {e}")

db_file = next((f for f in all_files if 'dry' in f.lower() and f.endswith('.csv')), None)
if db_file:
    df_db = parse_dry_bulb(os.path.join(search_path, db_file))
    df_db_clean = prepare_for_merge(df_db, 'dry_bulb')
    if not df_db_clean.empty:
        master_df = pd.merge(master_df, df_db_clean, on=['Station', 'Year', 'Month'], how='outer')
        print("Integrated: dry_bulb")

if master_df is not None:
    master_df.to_csv(output_step1, index=False)
    print("\n" + "="*50)
    print(f"STEP 1 COMPLETE! | Saved to: {output_step1}")
    print("="*50)