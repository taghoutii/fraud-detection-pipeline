# src/fraud_detection/data_loader.py
import pandas as pd
from pathlib import Path

def load_raw(raw_dir: str) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    train_txn = pd.read_csv(raw_dir / "train_transaction.csv")
    train_id = pd.read_csv(raw_dir / "train_identity.csv")
    df = train_txn.merge(train_id, on="TransactionID", how="left")
    return df

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes — IEEE-CIS has 400+ columns, this matters."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df