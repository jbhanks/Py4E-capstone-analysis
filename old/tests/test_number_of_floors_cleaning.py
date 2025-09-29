import pandas as pd
import numpy as np
import pytest

def clean_number_of_floors(series):
    # Step 1: Convert to string
    series = series.astype(str)
    # Step 2: Replace all missing value representations with np.nan
    series = series.replace(
        ['nan', 'NaN', 'None', 'NULL', '', ' ', '<NA>', None, pd.NA], np.nan
    )
    # Step 3: Convert to float (this will ensure all values are float or np.nan)
    series = pd.to_numeric(series, errors='coerce')
    # Step 4: Floor (or round) floats before casting to Int64
    series = np.floor(series)
    # Step 5: Now cast to pandas nullable integer type
    return series.astype('Int64')

def test_clean_number_of_floors():
    test_data = pd.Series([
        5, '7', 3.0, 'nan', 'NaN', None, pd.NA, '', ' ', '<NA>', 'NULL', 'None', np.nan, '12.0', '8.5', 'bad', 0
    ])
    expected = pd.Series([
        5, 7, 3, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 12, 8, pd.NA, 0
    ], dtype='Int64')
    result = clean_number_of_floors(test_data)
    # For non-numeric and non-missing, should be NA
    assert result.equals(expected), f"Expected {expected.tolist()}, got {result.tolist()}"

def test_clean_number_of_floors_all_missing():
    test_data = pd.Series(['nan', None, '', ' ', '<NA>', 'NULL', 'None', np.nan])
    expected = pd.Series([pd.NA] * len(test_data), dtype='Int64')
    result = clean_number_of_floors(test_data)
    assert result.equals(expected)

def test_clean_number_of_floors_all_valid():
    test_data = pd.Series([1, 2.0, '3', '4.0', 5])
    expected = pd.Series([1, 2, 3, 4, 5], dtype='Int64')
    result = clean_number_of_floors(test_data)
    assert result.equals(expected)

def test_clean_number_of_floors_with_floats():
    test_data = pd.Series([1.9, 2.1, 3.0, '4.7'])
    expected = pd.Series([1, 2, 3, 4], dtype='Int64')
    result = clean_number_of_floors(test_data)
    assert result.equals(expected)

if __name__ == "__main__":
    pytest.main([__file__])
