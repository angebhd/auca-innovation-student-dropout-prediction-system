import os
import pandas as pd
from src.data_processing import clean_student_data

RAW_PATH = "data/raw/students_raw.csv"
CLEANED_DIR = "data/cleaned"
CLEANED_PATH = os.path.join(CLEANED_DIR, "students_cleaned.csv")


def main():
   
    if not os.path.exists(CLEANED_DIR):
        os.makedirs(CLEANED_DIR)

   
    df_raw = pd.read_csv(RAW_PATH)
    print("Raw data shape:", df_raw.shape)

   
    df_cleaned, stats = clean_student_data(df_raw)

    print("Cleaning stats:", stats)
    print("Cleaned data shape:", df_cleaned.shape)

   
    df_cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEANED_PATH}")


if __name__ == "__main__":
    main()
