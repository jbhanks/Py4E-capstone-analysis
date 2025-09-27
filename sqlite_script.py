
import os
import subprocess
import re
import json
import codecs
import time
import dill
time.sleep(3)

from urllib.request import urlopen

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, event, Table, text, LargeBinary, ForeignKey
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import sessionmaker

from src.helpers import *
from src.dbutils import *
from src.ORMutils import *
from src.models import *
from src.geo import *
from src.pdfutils import *


# Load the environment
with open("environment_data/table_dicts.pkl", "rb") as f:
    env = dill.load(f)

# Restore the environment
globals().update(env)


import pprint
pprint.pprint(dataset_info_dict)


engine = create_engine(f'{SQLITE_PATH}?check_same_thread=False', echo=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def load_spatialite(dbapi_conn, connection_record):
    print("Loading SpatiaLite extension")
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension("mod_spatialite")
    dbapi_conn.enable_load_extension(False)


with engine.connect() as conn:
    print("Connection established")
    result = conn.execute(text("SELECT spatialite_version()"))
    spatialite_version = result.fetchone()
    print(f"SpatiaLite version: {spatialite_version[0]}")

# Enable WAL mode
with SessionLocal() as session:
    session.execute(text("PRAGMA journal_mode=WAL"))
    session.execute(text("PRAGMA synchronous = NORMAL"))
    session.execute(text("PRAGMA temp_store = MEMORY"))
    session.execute(text("PRAGMA wal_autocheckpoint=1000;"))  # Reduce I/O load
    session.execute(text("PRAGMA mmap_size = 30000000000;"))


# Initialize spatial metadata if not already present
# with engine.connect() as conn:
#     conn.execute(text("SELECT InitSpatialMetaData(1)"))
with SessionLocal() as session:
    result = session.execute(text("SELECT spatialite_version()"))
    spatialite_version = result.fetchone()
    print(f"SpatiaLite version: {spatialite_version[0]}")




with engine.connect() as conn:
    conn.execute(text("SELECT InitSpatialMetaData(1)"))


metadata = MetaData()
Base.metadata.reflect(bind=engine) 


multicolumns = {'zoning_district': 4, 'commercial_overlay': 2, 'special_purpose_district': 3}

for dataset in dataset_info_dict.values():
    for name,repetitions in multicolumns.items():
        print(name)
        print(f"Setting {name} columns")
        for k in dataset.col_customizations.keys():
            if dataset.col_customizations[k].new_name is None:
                dataset.col_customizations[k].new_name = dataset.col_customizations[k].short_name
        cols = {k:v for k,v in dataset.col_customizations.items() if dataset.col_customizations[k].new_name.startswith(name)}
        print(f'cols for {name} are: {cols}')
        main_col = [v for k,v in dataset.col_customizations.items() if dataset.col_customizations[k].new_name.endswith("_1")]


import pprint
pprint.pprint(dataset_info_dict)


# Import the MapPLUTO data from geo database file (.gdb)
pluto_version = "25v2_1"
gdb_path = f"{PROJECT_DATA}/files_to_use/MapPLUTO{pluto_version}.gdb"




geodata = {}
# List layers in the GDB file
layers = fiona.listlayers(gdb_path)
print("Layers in the GDB file:")
for layer in layers:
    print(layer)
    gdf = gpd.read_file(gdb_path, layer=layer)
    # gdf['borough'] = gdf['Borough'].replace(replacement_dict)
    try:
        gdf['wkb'] = gdf['geometry'].apply(lambda geom: geom.wkb if geom else None)
    except KeyError:
        pass
    geodata[layer] = gdf



gdf = geodata[f'MapPLUTO_{pluto_version}_clipped']
is_whole_number = {(gdf[col].notna() & (gdf[col] % 1 == 0)).all() for col in gdf.columns if gdf[col].dtype == 'float'}
gdf.columns



# Iterate over columns and change dtype to int where applicable
for col in gdf.columns:
    if  gdf[col].dtype == float and is_whole_number_series(gdf[col]):
        print(f'Column {col} is {is_whole_number_series(gdf[col])}')
        print(f'Converting {col} to integer')
        gdf[col] = gdf[col].astype('Int64')  # 'Int64' for nullable integer type in Pandas
    else:
        print(f"Skipping {col}")



from sqlalchemy import inspect
inspector = inspect(engine)
print(inspector.get_table_names())  # Ensure "basement_type_or_grade_lookup" is listed



col_customization_dict = dataset_info_dict['mapPLUTO'].col_customizations
rename_mappings = {v.short_name: v.new_name for v in col_customization_dict.values()}


gdf = gdf.rename(columns=rename_mappings)


print(gdf.columns)


# A few of the column names did not exactly match up due to slightly different field names than specified in the data dictionary, so these need to be renamed manually:

more_mappings = {
    "HealthCenterDistrict": "health_center_district",
    "SanitDistrict": "sanitation_district_number",
    "Sanitboro": "sanitation_district_boro",
    "FIRM07_FLAG": "2007_flood_insurance_rate_map_indicator",
    "PFIRM15_FLAG": "2015_preliminary_flood_insurance_rate_map",
}
gdf = gdf.rename(columns=more_mappings)


print(gdf.columns)


from sqlalchemy import Table, MetaData, Column, Integer, String, ForeignKey, LargeBinary, Float, Date
from sqlalchemy.orm import declarative_base

# Reflect the existing database tables once
metadata.reflect(bind=engine)

# Function to map custom dtype to SQLAlchemy types
def map_custom_dtype(dtype):
    if dtype == 'Integer':
        return Integer
    elif dtype == 'String':
        return String
    elif dtype == 'Float':
        return Float
    elif dtype == 'Date':
        return Date
    elif dtype == 'LargeBinary':
        return LargeBinary
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# Function to dynamically create the table class
def create_dynamic_table_class(table_name, col_customization_dict):
    attrs = {
        '__tablename__': table_name,
        'id': Column(Integer, primary_key=True, autoincrement=True),
        'geometry': Column(String),  
        'wkb': Column(LargeBinary),  # Use LargeBinary for WKB
        'Shape_Leng' : Column(Float), # Add columns not listed in the data dictionary
        'Shape_Area' : Column(Float),
    }
    attrs['__table_args__'] = {'extend_existing': True}
    
    for k, v in col_customization_dict.items():
        if any([name for name in multicolumns if name in k]):
            k = re.sub('_[0-9]$', '', k)
        col_type = map_custom_dtype(v.dtype)
        attrs[v.new_name] = Column(col_type)
    
    return type(table_name, (Base,), attrs)

# Create the MapPLUTO clipped table class
MapPLUTO_Clipped = create_dynamic_table_class(f'MapPLUTO_{pluto_version}_clipped', col_customization_dict)

# Reflect the metadata again to ensure it includes the new table class
metadata.reflect(bind=engine)

# Create all tables in the database
Base.metadata.create_all(engine)



from sqlalchemy.orm import sessionmaker
from shapely import wkb

# Create a session
session = SessionLocal()

# gdf = geodata['MapPLUTO_24v4_clipped']
def format_float(value):
    return str(value).rstrip('0').rstrip('.') if '.' in str(value) else str(value)

batch_size = 100000
with SessionLocal() as session:
    for start in range(0, len(gdf), batch_size):
        batch = gdf.iloc[start:start + batch_size]
        for idx, row in batch.iterrows():
            try:
                if row['apportionment_date']:
                    row['apportionment_date'] = parseDateString(row['apportionment_date'])
                for col in gdf.columns:
                    val = row[col]
                    if isinstance(val, pd.Series):
                        try:
                            first_value = row[col].iloc[0]
                            row[col] = first_value
                        except Exception as e:
                            print(f"Error processing Series in column {col} at row {idx}: {e}")
                    # Replace NA values with None so that SQLAlchemy inserts them as NULL:
                    if pd.isna(val):
                        row[col] = None
                # Prepare the geometry and entry object
                geometry_wkb = row['geometry'].wkb if row['geometry'] else None
                pluto_entry = MapPLUTO_Clipped(
                    geometry=geometry_wkb,
                    **{col: row[col] for col in gdf.columns if col not in ['geometry']}
                )
                session.add(pluto_entry)
            except Exception as e:
                print(f"Error at row index {idx}")
                for col in gdf.columns:
                    try:
                        print(f"Column: {col}, Value: {row[col]}, Type: {type(row[col])}")
                    except Exception as sub_e:
                        print(f"Error printing column {col}: {sub_e}")
                raise e  # re-raise after logging for further debugging
        session.commit()



import pprint
pprint.pprint(dataset_info_dict)


import geopandas as gpd
import pandas as pd
from shapely import wkb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from sqlalchemy import create_engine, event, text

# Read the data from the database
query = f"SELECT zip_code, geometry FROM MapPLUTO_{pluto_version}_clipped"
df = pd.read_sql(query, engine)

# Debug: Print the DataFrame columns
print("DataFrame columns:", df.columns)

# Convert the geometry column from WKB to Shapely geometries
df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if x else None)

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Print the GeoDataFrame
print(gdf.head())

# Ensure that zip_code is preserved during the dissolve process
merged_gdf = gdf.dissolve(by='zip_code', aggfunc={'zip_code': 'first'})  # Explicit aggregation of zip_code

# Check if zip_code is now present after dissolving
print(merged_gdf.columns)  # Should include 'zip_code'

# Create a new adjacency graph based on the merged geometries
G = nx.Graph()

# Add nodes and edges based on adjacency of merged shapes
for i, shape1 in merged_gdf.iterrows():
    for j, shape2 in merged_gdf.iterrows():
        if i != j and shape1.geometry.touches(shape2.geometry):
            G.add_edge(i, j)

# Perform graph coloring to ensure adjacent shapes don't share the same color
color_map = nx.coloring.greedy_color(G, strategy="largest_first")

# Plot the map with the colors assigned
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Normalize the color map to cover the full range of the node indices
norm = mcolors.Normalize(vmin=min(color_map.values()), vmax=max(color_map.values()))
sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20, norm=norm)

# Color the merged geometries based on the graph coloring using the full palette
merged_gdf['color'] = merged_gdf.index.map(color_map)
merged_gdf.plot(ax=ax, color=[sm.to_rgba(i) for i in merged_gdf['color']], edgecolor='black', linewidth=0, legend=False)

# Add labels at the center of each merged shape
for _, row in merged_gdf.iterrows():
    centroid = row.geometry.centroid
    ax.text(centroid.x, centroid.y, str(row['zip_code']), fontsize=2, ha='center', va='center')

# Add a colorbar to visualize the full range of colors
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Color Range (Graph Coloring)', rotation=270, labelpad=20)

plt.savefig(f"{PROJECT_DATA}/figures/map_output_zip_shuffled2.pdf", format="pdf")

plt.show()


datatype_mappings = {"meta_data" : String, "calendar_date" : Date, "number" : Float, "text" : String, "point" : String}


import jsonlines
import orjson
import time
from shapely import from_wkt  # Vectorized conversion function in Shapely 2.0
from geoalchemy2.shape import from_shape
from sqlalchemy.engine import Engine
import pandas as pd
from sqlalchemy import text


def split_address(row_data, address_col):
    if address_col in row_data.keys():
        if row_data[address_col] is None:
            row_data["building_num"], row_data["street"] = None, None
            row_data.pop(address_col)
            return row_data
        else:
            if row_data[address_col][0].isdigit():
                try:
                    addr = row_data[address_col].split(" ", 1)
                    if len(addr) == 1:
                        addr = [None] + addr
                    row_data["building_num"], row_data["street"] = addr
                except Exception as e:
                    print(e)
                    print(row_data[address_col])
            else:
                row_data["building_num"], row_data["street_name"] = (
                    None,
                    row_data[address_col],
                )
        row_data.pop(address_col)
    return row_data

def convert_wkt(rows_to_insert):
    # Batch convert geometries.
    raw_wkts = [r.get('_raw_geocoded_column') for r in rows_to_insert]
    try:
        shapely_geoms = from_wkt(raw_wkts)
    except Exception as e:
        print(f"Error converting batch geometry at row {idx}: {e}")
        shapely_geoms = [None] * len(rows_to_insert)
    geoms = [
        from_shape(geom, srid=4326) if geom is not None else None 
        for geom in shapely_geoms
    ]
    for r, geom in zip(rows_to_insert, geoms):
        r['geocoded_column'] = geom
        r.pop('_raw_geocoded_column', None)



import pprint
pprint.pprint(dataset_info_dict)


dataset = dataset_info_dict['tax_liens']
pprint.pprint(dataset)


def insert_dataset(engine, dataset, jsonfile, columns, batch_size=10000):
    """
    Process the JSON Lines file in batches using pandas vectorized operations.
    Uses engine.begin() to leverage executemany().
    """
    col_names = list(columns.keys())
    expected_width = len(col_names)
    rows_buffer = []
    insert_stmt = None
    DynamicTable = None

    def custom_loads(s):
        return orjson.loads(s.encode("utf-8"))

    def sanitize_rows(rows):
        cleaned = []
        for i, row in enumerate(rows):
            new_row = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    print(f"Offending value in row {i}, column '{k}': {repr(v)}")
                    v = None
                new_row[k] = v
            cleaned.append(new_row)
        return cleaned

    def validate_no_nans(rows):
        for i, row in enumerate(rows):
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    print(f"❌ Still found NaN/Inf at row {i}, column '{k}' — value: {repr(v)}")
                    raise ValueError("NaN or Inf sneaked through")

    def process_batch(df, nullable_integer_columns):
        df = df.where(pd.notnull(df), None)

        # Sanitize strings
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
                print(f"⚠️ Warning: {col} contains NaN")

        # Fix nullable INTEGER columns: convert to nullable Int64
        for col in nullable_integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Dates
        datetime_cols = [key for key in columns if columns[key] is Date]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].apply(lambda x: x.date() if pd.notnull(x) else None)
        if 'geocoded_column' in df.columns:
            df['_raw_geocoded_column'] = df['geocoded_column']
            df['geocoded_column'] = None

        df = split_address(df, 'address')

        rows = df.to_dict(orient="records")
        rows = sanitize_rows(rows)
        convert_wkt(rows)
        validate_no_nans(rows)
        return rows

    with jsonlines.open(jsonfile, mode='r', loads=custom_loads) as reader:
        batch_start = time.perf_counter()
        for idx, row in enumerate(reader):
            if isinstance(row, list) and len(row) > expected_width:
                row = row[:expected_width]
            rows_buffer.append(row)

            if (idx + 1) % batch_size == 0:
                df = pd.DataFrame(rows_buffer, columns=col_names)
                t0 = time.perf_counter()
                if insert_stmt is None:
                    print(f"📐 Creating table for dataset {dataset.short_name}")
                    DynamicTable = create_table_for_dataset(
                        columns=dataset.col_types,
                        prefix=dataset.short_name,
                        engine=engine
                    )
                    insert_stmt = DynamicTable.__table__.insert().prefix_with("OR IGNORE")
                    print("📋 Table schema:")
                    for col in DynamicTable.__table__.columns:
                        print(f"  {col.name}: {col.type}, nullable={col.nullable}")
                    nullable_integer_columns = [
                        col for col, typ in columns.items()
                        if typ is Integer and DynamicTable.__table__.columns[col].nullable
                    ]
                batch_rows = process_batch(df, nullable_integer_columns)
                t2 = time.perf_counter()
                with engine.begin() as conn:
                    conn.execute(insert_stmt, batch_rows)
                t3 = time.perf_counter()
                print(f"📤 Inserted batch in {t3 - t2:.2f}s, total time {t3 - batch_start:.2f}s")

                rows_buffer.clear()
                batch_start = time.perf_counter()

        # Final flush
        if rows_buffer:
            DynamicTable = create_table_for_dataset(
                columns=dataset.col_types,
                prefix=dataset.short_name,
                engine=engine
            )
            insert_stmt = DynamicTable.__table__.insert().prefix_with("OR IGNORE")
            print(f'insert_stmt: {insert_stmt}')
            nullable_integer_columns = [
                col for col, typ in columns.items()
                if typ is Integer and DynamicTable.__table__.columns[col].nullable]
            print(f'nullable_integer_columns: {nullable_integer_columns}')
            df = pd.DataFrame(rows_buffer, columns=col_names)
            print(df.head())
            batch_rows = process_batch(df, nullable_integer_columns)
            with engine.begin() as conn:
                conn.execute(insert_stmt, batch_rows)
            print("✅ Final batch inserted successfully.")


for name, dataset in dataset_info_dict.items():
    if dataset.format == 'json':
        print(f'Starting dataset {dataset.short_name}')
        print(f'The dataset to be processed is {dataset}')
        print(f'Starting dataset {dataset.short_name}')
        insert_dataset(engine, dataset, dataset.dataset_path, dataset.col_types, batch_size=10000)






