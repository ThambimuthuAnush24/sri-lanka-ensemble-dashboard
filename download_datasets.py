"""Download and combine REAL Sri Lankan dengue district/week observations.
Run: python download_datasets.py (requires internet)
"""
from pathlib import Path
import pandas as pd

HISTORICAL_URL = "https://raw.githubusercontent.com/devmividanage-hub/Dengue-Risk-MLOps-1/master/data/raw/dengue/sri_lanka_dengue_weekly.csv"
RECENT_URL = "https://raw.githubusercontent.com/dhananjaya-hbc/lanka-data-tracker/main/data/dengue.csv"
DEST = Path(__file__).resolve().parent / "data"

def clean(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(".", "_") for c in df.columns]
    df = df.rename(columns={"start_date": "week_start", "end_date": "week_end"})
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["district"] = df["district"].str.strip().replace({"NuwaraEliya":"Nuwara Eliya", "Hambanthota":"Hambantota", "Kalmune":"Kalmunai"})
    df["cases"] = pd.to_numeric(df["cases"],errors="coerce")
    return df[["year","week","week_start","district","cases"]].dropna()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    historic = pd.read_csv(HISTORICAL_URL)
    recent = pd.read_csv(RECENT_URL)
    h, r = clean(historic), clean(recent)
    df = (pd.concat([h,r],ignore_index=True).drop_duplicates(["year","week","district"],keep="last")
          .sort_values(["week_start","district"]).reset_index(drop=True))
    df.to_csv(DEST / "sri_lanka_dengue_combined.csv",index=False)
    print(f"Historic rows: {len(h):,}; recent rows: {len(r):,}")
    print(f"Combined unique district-week rows: {len(df):,}")
    print(f"Saved to: {DEST / 'sri_lanka_dengue_combined.csv'}")
    print("NOTE: Historic data provenance/licensing is unverified by the GitHub repository.")

if __name__ == "__main__":
    main()
