import pandas as pd
import numpy as np

def handle_missing(df: pd.DataFrame, num_thresh: float = 0.9) -> pd.DataFrame:
    """Drop columns missing > num_thresh; impute remaining."""
    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > num_thresh].index.tolist()
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = df[num_cols].fillna(-999)   # tree models handle sentinel values well
    df[cat_cols] = df[cat_cols].fillna("missing")
    return df

def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Frequency encoding — robust to high cardinality (e.g. card1: 13k+ unique values),
    avoids the dimensionality blowup of one-hot encoding on this dataset."""
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq_enc"] = df[col].map(freq)
    return df.drop(columns=cat_cols)