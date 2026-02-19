from src.data_generation import generate_student_dummy_data
import os
import pandas as pd
import yaml

try:
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    output_dir = cfg['paths']['raw_dir']
except Exception:
    output_dir = "data/raw"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        df = generate_student_dummy_data(n_samples=400,seed=42)
        result_path = os.path.join(output_dir,"raw_student.csv")
        df.to_csv(result_path, index=False)
        print(f"Generated {len(df)} students at {result_path}")