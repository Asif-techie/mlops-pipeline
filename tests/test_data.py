import pytest
from src.data import load_data

def test_load_data_not_empty():
    df = load_data()
    assert df is not None
    assert not df.empty
    assert "num" in df.columns
