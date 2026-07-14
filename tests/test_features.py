import pandas as pd
from src.fraud_detection.features import add_time_since_last_txn

def test_time_since_last_txn_no_leakage():
    df = pd.DataFrame({
        "card1": [1, 1, 1],
        "TransactionDT": [100, 200, 500],
    })
    out = add_time_since_last_txn(df, "card1")
    assert out.iloc[0][f"card1_time_since_last"] == -1   # first txn has no history
    assert out.iloc[1][f"card1_time_since_last"] == 100