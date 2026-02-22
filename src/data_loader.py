import pandas as pd

def load_csv(path: str):
    """
    Load a CSV dataset and return a DataFrame
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df