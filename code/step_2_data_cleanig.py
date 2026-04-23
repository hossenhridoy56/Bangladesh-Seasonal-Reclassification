import pandas as pd
import os

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

input_dir = os.path.join(root_folder, "data", "processed")
input_path = os.path.join(input_dir, "master_step1_raw_integrated.csv")
output_step2 = os.path.join(input_dir, "master_step2_cleaned.csv")

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

print(f"--- Step 2: Data Cleaning & Filtering ---")

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found!")
else:
    df = pd.read_csv(input_path)

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[(df['Year'] >= 1983) & (df['Year'] <= 2014)]

    df['Station'] = df['Station'].astype(str).str.replace(r'[^a-zA-Z]', '', regex=True).str.strip().str.title()
    
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

    df = df.sort_values(['Station', 'Year', 'Month']).reset_index(drop=True)

    vars_7 = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']
    df = df.dropna(subset=vars_7, how='all')

    df.to_csv(output_step2, index=False)

    print("\n" + "="*50)
    print(f"STEP 2 SUCCESSFUL!")
    print(f"Saved to: {output_step2}")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Stations: {df['Station'].nunique()}")
    print("="*50)