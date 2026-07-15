import pandas as pd
import numpy as np

def handle_missing(df: pd.DataFrame, num_thresh: float = 0.9) -> pd.DataFrame:
    #Drop columns missing > num_thresh; impute remaining
    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > num_thresh].index.tolist()
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = df[num_cols].fillna(-999)   # tree models learn that -999 is a missing value
    df[cat_cols] = df[cat_cols].fillna("missing")
    return df

def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    #Frequency encoding for robust to high cardinality (card1: 13k+ unique values)
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True) #noramlized returns percentage instead of counts
        df[col + "_freq_enc"] = df[col].map(freq)
    return df.drop(columns=cat_cols)

def time_based_split(df: pd.DataFrame, time_col: str, test_size: float = 0.2):
    df = df.sort_values(time_col)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test