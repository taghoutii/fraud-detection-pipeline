import pandas as pd
from src.fraud_detection.preprocessing import handle_missing, encode_categoricals

def test_handle_missing_drops_high_missing_cols():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
    out = handle_missing(df, num_thresh=0.5)
    assert "a" not in out.columns

def test_encode_categoricals_creates_freq_col():
    df = pd.DataFrame({"cat": ["x", "x", "y"]})
    out = encode_categoricals(df, ["cat"])
    assert "cat_freq_enc" in out.columns
    assert "cat" not in out.columns