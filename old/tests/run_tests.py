
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import fiona
import pytest

# Import the cleaning function from your other test file
from test_number_of_floors_cleaning import clean_number_of_floors

def test_number_of_floors_from_gdb():
    # Set up the path to your GDB file and layer name
    PROJECT_DATA = os.environ.get("PROJECT_DATA", "./data")
    pluto_version = "25v1_1"
    gdb_path = f"/mnt/Datasets/PROJECTDATA/nyc_real_estate_data/files_to_use/MapPLUTO25v1_1.gdb"

    # Find the correct layer
    layers = fiona.listlayers(gdb_path)
    # Use the first layer or specify the correct one
    layer = [l for l in layers if "clipped" in l.lower()]
    if layer:
        layer = layer[0]
    else:
        layer = layers[0]

    gdf = gpd.read_file(gdb_path, layer=layer)

    # Check that the column exists (try both original and renamed)
    col = None
    for candidate in ["number_of_floors", "NumFloors", "NUM_FLOORS"]:
        if candidate in gdf.columns:
            col = candidate
            break
    assert col is not None, f"No number_of_floors column found in {gdf.columns.tolist()}"

    # Run the cleaning function
    cleaned = clean_number_of_floors(gdf[col])

    # Check that the result is of Int64 dtype
    assert pd.api.types.is_integer_dtype(cleaned.dtype)
    # Check that all missing/invalid values are pd.NA or np.nan
    # (Optional) Check that all valid values are integers
    valid = cleaned.dropna()
    assert (valid == valid.astype(int)).all()

    # Optionally print some stats
    print("Cleaned number_of_floors value counts:")
    print(cleaned.value_counts(dropna=False).head())

if __name__ == "__main__":
    pytest.main([__file__])