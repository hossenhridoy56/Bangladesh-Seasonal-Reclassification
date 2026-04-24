import pandas as pd
import os

current_script_path = os.path.abspath(__file__)
root_folder = os.path.dirname(os.path.dirname(current_script_path))

input_dir = os.path.join(root_folder, "data", "processed")
input_path = os.path.join(input_dir, "master_step2_cleaned.csv")
output_step3 = os.path.join(input_dir, "FINAL_RESEARCH_READY_DATA.csv")

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

vars_7 = ['rainfall', 'tmax', 'tmin', 'cloud', 'wind', 'humidity', 'dry_bulb']

print(f"--- Step 3: Missing Value Estimation & Statistics ---")

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found!")
else:
    df = pd.read_csv(input_path)

    print("Processing missing values...")
    for v in vars_7:
        if v in df.columns:
            df[v] = df.groupby(['Station', 'Month'], observed=False)[v].transform(lambda x: x.fillna(x.mean()))
            df[v] = df.groupby('Month', observed=False)[v].transform(lambda x: x.fillna(x.mean()))

    df = df.fillna(df.mean(numeric_only=True))

    df.to_csv(output_step3, index=False)

    print("\n" + " FINAL DATASET SUMMARY ".center(60, "="))
    
    stats = df[vars_7].describe().T[['mean', 'std', 'min', 'max']]
    stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
    
    table_dir = os.path.join(root_folder, "tables")
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)
        
    table_path = os.path.join(table_dir, "Table1_General_Statistics.csv")
    
    stats.round(2).to_csv(table_path, index=True) 

    print("\nGeneral Statistics (1983-2014):")
    print(stats.round(2).to_string())

    print("\n" + "-"*60)
    print(f"Total Period:         1983 - 2014")
    print(f"Total Stations:       {df['Station'].nunique()}")
    print(f"Total Monthly Records: {len(df)}")
    print(f"Missing Values:       0 (Fully Imputed)")
    print(f"Final Output File:    {output_step3}")
    print("="*60)