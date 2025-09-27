from sqlalchemy import Engine
from sqlalchemy import (
    inspect,
    Column,
    Integer,
    Numeric,
    BigInteger,
    SmallInteger,
    String,
    Float,
    Date,
    JSON,
    MetaData,
    event,
    Table,
    text,
    LargeBinary,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.sqlite import insert
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape


import re
import datetime
import pandas as pd
import orjson
import jsonlines
from .helpers import merge_dicts, parseDateString, split_address, textClean

from src.models import ColCustomization



class Base(DeclarativeBase):
    pass


def table_exists(engine: Engine, table_name: str, schema: str = None) -> bool:
    """
    Check if a table exists in the database.

    Args:
        engine (Engine): The SQLAlchemy engine connected to the database.
        table_name (str): The name of the table to check.
        schema (str, optional): The schema name, if applicable. Defaults to None.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    inspector = inspect(engine)
    return table_name in inspector.get_table_names(schema=schema)


def write_layer_to_db(layer_dict, layer_name, orm_classes, session):
    gdf = layer_dict[layer_name]
    # Prepare the data for insertion
    batch_size = 100
    for start in range(0, len(gdf), batch_size):
        batch = gdf.iloc[start : start + batch_size]
        for _, row in batch.iterrows():
            if "geometry" in gdf.columns:
                geometry_wkb = row["geometry"].wkb if row["geometry"] else None

            # Convert date fields to Python date objects
            for date_field in ["YearBuilt", "YearAlter1", "YearAlter2", "APPDate"]:
                if pd.notnull(row[date_field]) and row[date_field] != 0:
                    try:
                        # Handle year-only format
                        row[date_field] = datetime.datetime.strptime(
                            str(int(row[date_field])), "%Y"
                        ).date()
                    except ValueError:
                        try:
                            # Handle full date format
                            row[date_field] = datetime.datetime.strptime(
                                str(row[date_field]), "%m/%d/%Y"
                            ).date()
                        except ValueError:
                            row[date_field] = None
                else:
                    row[date_field] = None

            if "geometry" in gdf.columns:
                entry = orm_classes[layer_name.capitalize()](
                    geometry=geometry_wkb,
                    **{col: row[col] for col in gdf.columns if col not in ["geometry"]},
                )
            else:
                entry = orm_classes[layer_name.capitalize()](
                    **{col: row[col] for col in gdf.columns if col not in ["geometry"]}
                )
            session.add(entry)
        session.commit()


def create_lookup_table(metadata, lookup_table_name, text_column_name):
    table_name = f"{lookup_table_name}_lookup"

    if table_name in metadata.tables:  # Avoid redefining existing tables
        print(f"Skipping existing table: {table_name}")
        return

    lookup_table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column(
            text_column_name, String, unique=True, nullable=False, default="NO DATA"
        ),
        Column("info", String, unique=False, nullable=True, default=None),
    )
    return lookup_table


def set_column(colname, dtype):
    return Column(
        colname,
        dtype,
        primary_key=False,
        nullable=True,
        unique=False,
        index=True,
        # No server_default for “NULL” — just leave it nullable.
        autoincrement=False,   # only PKs should autoincrement
    )

# def create_table(columns, engine, tabname):
#     columns = {'local_id': Column(Integer, primary_key=True)} | {colname: set_column(colname, dtype) for colname,dtype in columns.items() if isinstance(colname, str)}
#     columns['sid'] = Column(String, unique=True, nullable=False)
#     table_class = type(f"{tabname.capitalize()}Table", (Base,), {
#         "__tablename__": f"{tabname}",
#         "__table_args__": {'extend_existing': True},
#         **columns
#     })
#     print(f'Creating table class {tabname}: {table_class}')
#     # Create the table in the database
#     Base.metadata.create_all(engine)
#     return table_class

from sqlalchemy import Table
from sqlalchemy.orm import declarative_base

# Assuming you already created Base with a naming convention (recommended)
# Base = declarative_base(metadata=metadata)

def create_table(columns, engine, tabname):
    # If the table already exists in metadata, reuse it instead of redefining
    if tabname in Base.metadata.tables:
        table = Base.metadata.tables[tabname]
        # ✅ ensure it exists physically
        table.create(bind=engine, checkfirst=True)
        return type(f"{tabname.capitalize()}Table", (Base,), {"__table__": table})

    # First definition: build columns once
    cols = {'local_id': Column(Integer, primary_key=True)}
    cols.update({
        colname: set_column(colname, dtype)
        for colname, dtype in columns.items() if isinstance(colname, str)
    })
    cols['sid'] = Column(String, unique=True, nullable=False)

    table_class = type(f"{tabname.capitalize()}Table", (Base,), {
        "__tablename__": tabname,
        "__table_args__": {"keep_existing": True},
        **cols
    })

    # Create just this table (idempotent)
    table_class.__table__.create(bind=engine, checkfirst=True)
    return table_class


# def create_table_for_dataset(columns, prefix, engine):
#     tabname = re.sub('.*/', '', prefix)
#     print(f'Creating table {tabname}')
#     DynamicTable = create_table(columns, engine, tabname)
#     if DynamicTable is None:
#         raise(Exception("DynamicTable not made!"))
#     return DynamicTable

from sqlalchemy import Column, Integer, String, Float, Date
from geoalchemy2 import Geometry

# def create_table_for_dataset(columns, prefix, engine):
#     """
#     Dynamically build a SQLAlchemy ORM class with PostGIS geometry columns.
#     - `columns`: dict of column_name -> dtype (from dataset.col_types)
#     - `prefix`: table name (usually dataset.short_name)
#     - `engine`: SQLAlchemy engine
#     """
#     tabname = re.sub(".*/", "", prefix)
#     print(f"Creating table {tabname}")

#     # Reuse existing table class if already present
#     if tabname in Base.metadata.tables:
#         table = Base.metadata.tables[tabname]
#         table.create(bind=engine, checkfirst=True)
#         return type(f"{tabname.capitalize()}Table", (Base,), {"__table__": table})

#     # Build columns
#     cols = {
#         "id": Column(Integer, primary_key=True, autoincrement=True)
#     }
#     for colname, dtype in columns.items():
#         try:
#             cols[colname] = Column(map_postgres_dtype(dtype))
#         except ValueError as e:
#             print(f"⚠️ Skipping column {colname}: {e}")

#     # Define dynamic ORM class
#     table_class = type(
#         f"{tabname.capitalize()}Table",
#         (Base,),
#         {
#             "__tablename__": tabname,
#             "__table_args__": {"extend_existing": True},
#             **cols,
#         },
#     )

#     # Actually create table in the database
#     Base.metadata.create_all(engine, tables=[table_class.__table__])
#     return table_class

def create_table_for_dataset(columns, prefix, engine):
    """
    Dynamically build a SQLAlchemy ORM class for Postgres + PostGIS.
    Uses map_column_types() to resolve all column dtypes.
    """
    tabname = re.sub(r".*/", "", prefix)
    print(f"Creating table {tabname}")

    # If table already exists in metadata, reuse it
    if tabname in Base.metadata.tables:
        table = Base.metadata.tables[tabname]
        table.create(bind=engine, checkfirst=True)
        return type(f"{tabname.capitalize()}Table", (Base,), {"__table__": table})

    # Map dataset dtypes → SQLAlchemy/GeoAlchemy types
    mapped_cols = map_column_types(columns)

    # Build column dict
    cols = {
        "id": Column(Integer, primary_key=True, autoincrement=True)
    }
    for colname, coltype in mapped_cols.items():
        cols[colname] = Column(coltype)

    # Define dynamic ORM class
    table_class = type(
        f"{tabname.capitalize()}Table",
        (Base,),
        {
            "__tablename__": tabname,
            "__table_args__": {"extend_existing": True},
            **cols,
        },
    )

    # Create table in the DB (idempotent)
    Base.metadata.create_all(engine, tables=[table_class.__table__])
    return table_class



def populate_lookup_table(engine, lookup_table, source_table_name, lookup_table_name, text_column_name, chunk_size=1000, max_retries=5, drop_old_columns=False):
    """
    Populate a lookup table in chunks with retries for database lock issues.
    """
    def retry(func, *args, **kwargs):
        """Retry function with backoff for SQLite locks."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except OperationalError as e:
                if "database is locked" in str(e):
                    print(f"Database is locked. Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(0.2 * (attempt + 1))  # Gradual backoff
                else:
                    raise
        raise Exception("Exceeded maximum retries due to database locks.")
    
    with engine.connect() as connection:
        # Ensure the new column exists
        try:
            retry(connection.execute, text(f"ALTER TABLE {source_table_name} ADD COLUMN {text_column_name}_id INTEGER"))
        except Exception as e:
            print(f"Column creation skipped or failed: {e}")

        # Process unique values in chunks
        unique_query = f"SELECT DISTINCT {text_column_name} FROM {source_table_name}"
        unique_values_iter = pd.read_sql(unique_query, engine, chunksize=chunk_size)
        
        all_unique_values = []
        for chunk in unique_values_iter:
            all_unique_values.extend(chunk[text_column_name].dropna().tolist())

        # Insert into the lookup table in batches
        unique_values = list(set(all_unique_values))
        for i in range(0, len(unique_values), chunk_size):
            batch_values = unique_values[i:i + chunk_size]
            stmt = insert(lookup_table).values([{'name_or_code': value} for value in batch_values]).on_conflict_do_nothing()
            try:
                retry(connection.execute, stmt)
            except Exception as e:
                print(f"Error inserting batch: {e}")

        # Update the source table with foreign key references
        update_stmt = text(f"""
        UPDATE {source_table_name}
        SET {text_column_name}_id = (
            SELECT id 
            FROM {lookup_table_name}
            WHERE name_or_code = {source_table_name}.{text_column_name}
        )
        """)
        try:
            retry(connection.execute, update_stmt)
        except Exception as e:
            print(f"Error updating foreign keys: {e}")
        connection.commit()
        # Remove the original text column (optional)
        if drop_old_columns:
            print(f"Dropping old column {text_column_name} from {source_table_name}...")
            connection.execute(text(f"ALTER TABLE {source_table_name} DROP COLUMN {text_column_name}"))
        connection.commit()

def retry(func, *args, max_retries=3, **kwargs):
    """Retry function with backoff for SQLite locks."""
    for attempt in range(max_retries):
        try:
            print(
                f"Retry attempt {attempt + 1}: func={func}, args={args}, kwargs={kwargs}"
            )
            return func(*args, **kwargs)
        except OperationalError as e:
            if "database seems to be locked" in str(e):
                print(f"Database is locked. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1 * (attempt + 1))  # Gradual backoff
            else:
                raise
    raise Exception("Exceeded maximum retries due to database locks.")


################################################3
############################
def is_number(value):
    """Check if a string represents a number (integer or float)."""
    try:
        float(value)
        return True
    except ValueError:
        return False


# It would be better not to hard code these mappings, but to infer them from the data.
# However, that would require scanning the entire dataset first to determine the types.
# For now, we will use a heuristic based on column names and provided types.
from sqlalchemy import Integer, BigInteger, String, Float, Date, JSON, Numeric
from geoalchemy2 import Geometry

def map_column_types(column_types):
    """Map column names to SQLAlchemy/GeoAlchemy2 types for Postgres + PostGIS."""
    geom_types = {
        "point": Geometry(geometry_type="POINT", srid=4326),
        "multipoint": Geometry(geometry_type="MULTIPOINT", srid=4326),
        "linestring": Geometry(geometry_type="LINESTRING", srid=4326),
        "multilinestring": Geometry(geometry_type="MULTILINESTRING", srid=4326),
        "polygon": Geometry(geometry_type="POLYGON", srid=4326),
        "multipolygon": Geometry(geometry_type="MULTIPOLYGON", srid=4326),
    }

    value_cols = {
        "fullval", "avland", "avtot", "exland", "exempttot",
        "avland2", "avtot2", "exland2", "extot2"
    }

    return {
        column: (
            Numeric(14, 2) if column in value_cols else
            Integer if dtype == "number"
                      or column in ["created_at", "updated_at"]
                      or column.startswith("int") else
            String if dtype == "text"
                      or column == "sid"
                      or column.startswith("str") else
            Date if dtype == "calendar_date" else
            Float if isinstance(dtype, str) and dtype.startswith("float") else
            JSON if dtype == "location" else
            geom_types.get(dtype.lower()) if isinstance(dtype, str) and dtype.lower() in geom_types else
            dtype
        )
        for column, dtype in column_types.items()
        if not column.startswith(":@")
    }


# def map_column_types(column_types):
#     """Map column names to SQLAlchemy types based on the given column types."""
#     return {
#         column: (
#             Integer if dtype == "number" or column in ['created_at', 'updated_at'] or column.startswith('int') else
#             BigInteger if column in ['fullval', 'avland', 'avtot', 'exland', 'exempttot', 'avland2', 'avtot2', 'exland2', 'extot2'] else
#             String if dtype == "text" or column == "sid" or column.startswith('str') else
#             Date if dtype == "calendar_date" else
#             Float if dtype.startswith('float') else
#             JSON if dtype == "location" else
#             Geometry(geometry_type='POINT', srid=4326) if dtype == "point" else
#             Geometry(geometry_type='MULTIPOLYGON', srid=4326) if dtype == "multipolygon" else
#             (Geometry(geometry_type='POLYGON', srid=4326) if dtype == "polygon" else dtype)
#         )
#         for column, dtype in column_types.items() if not column.startswith(':@')
#     }

def convert_row_values(row, column_types, column_names):
    """Convert the values in the row based on their data types."""
    for col, dtype in column_types.items():
        if col not in row:
            continue
        if dtype == "Integer":
            row[col] = int(row[col]) if row[col] is not None else None
        elif dtype == "Float":
            row[col] = float(row[col]) if row[col] is not None else None
        elif dtype == "Date":
            row[col] = pd.to_datetime(row[col]) if row[col] is not None else None
    return row

# def update_numeric_columns(row, column_names, numeric_indices, new_column_types):
#     """Check numeric columns and update the column types accordingly."""
#     row_list = [row.get(col) for col in column_names]
#     for idx in numeric_indices:
#         value = row_list[idx]
#         if value is None:
#             continue  # Skip null values
#         if isinstance(value, (int, float)):
#             if float(value) % 1 != 0:
#                 colname = column_names[idx]
#                 new_column_types[colname] = Float
#                 # break
#         elif isinstance(value, str):
#             val_str = value.strip()
#             if not val_str.isdigit() and is_number(val_str):
#                 if float(val_str) % 1 != 0:
#                     colname = column_names[idx]
#                     new_column_types[colname] = Float
#                     # break
#         else:
#             print(f"Unexpected type for numeric column {column_names[idx]}: {type(value)}")
#     return new_column_types

import re
NUMERIC_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def update_numeric_columns(row, column_names, numeric_indices, new_column_types):
    """
    If a column was declared "number" but we encounter:
      - a non-numeric token (e.g., '25v2.1', 'E-61') -> String
      - a numeric with decimals                        -> Float
      - a code-like numeric (leading zeros or codey name) -> String
    """
    CODEY_COL_TOKENS = (
        "zip", "zip_code",
        "census", "tract", "block",
        "district", "school_district", "police_precinct",
        "tax_map", "tax_map_num",
        "sanitation_district_number",
        "version", "version_number",
        "designation", "e-designation",  # <-- catches e-designation_number
        "fire_company",
    )

    def looks_code_like(colname: str, val_str: str) -> bool:
        if val_str.startswith("0") and len(val_str) > 1:
            return True
        name = colname.lower()
        return any(tok in name for tok in CODEY_COL_TOKENS)

    row_list = [row.get(col) for col in column_names]

    for idx in numeric_indices:
        colname = column_names[idx]
        value = row_list[idx]

        if value is None or value == "":
            continue

        # If we already promoted to String or Float earlier in the scan, don't downgrade.
        tname = getattr(new_column_types.get(colname), "__name__", str(new_column_types.get(colname)))
        if tname in ("String", "Text"):
            continue

        if isinstance(value, (int, float)):
            # floats with a fractional part => Float
            if isinstance(value, float) and value % 1 != 0:
                new_column_types[colname] = Float
            continue

        if isinstance(value, str):
            val_str = value.strip()

            # non-numeric token (letters, hyphens, etc.) => String
            if not NUMERIC_RE.match(val_str):
                new_column_types[colname] = String
                continue

            # numeric string with a fractional component => Float
            try:
                fv = float(val_str)
                if "." in val_str or "e" in val_str.lower():
                    # keep Integer only if it’s an exact int
                    if not fv.is_integer():
                        new_column_types[colname] = Float
                        continue
            except Exception:
                new_column_types[colname] = String
                continue

            # code-like columns => String to preserve formatting/semantics
            if looks_code_like(colname, val_str):
                new_column_types[colname] = String
                continue

    return new_column_types


def set_dtypes(filename, column_types):
    """Main function to process the file and set data types."""
    print(f'Starting {filename}')
    if not filename.endswith('.json'):
        print(f'{filename} is not a json file, skipping.')
        return column_types

    # Build the initial mapping of column names to SQLAlchemy types.
    new_column_types = map_column_types(column_types)
    print(f'Initial new_column_types: {new_column_types}')

    # Preserve the order of columns.
    column_names = list(column_types.keys())
    print(f'column_names: {column_names}')
    numeric_indices = [
        idx for idx, col in enumerate(column_names)
        if column_types[col] == "number"
    ]

    # Open the file via jsonlines.
    with jsonlines.open(filename, mode='r') as reader:
        for row in reader:
            # If each row comes as a list, convert it into a dict using the ordering in column_names.
            if isinstance(row, list):
                row = {col_name: (row[i] if i < len(row) else None)
                       for i, col_name in enumerate(column_names)}

            # Convert the values for known types.
            row = convert_row_values(row, column_types, column_names)

            # Update numeric column types.
            new_column_types = update_numeric_columns(row, column_names, numeric_indices, new_column_types)

    return new_column_types


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



# def insert_dataset(session, engine, dataset, jsonfile, columns, batch_size=10000, commit_interval=100):
#     """
#     Process the JSON Lines file in batches using pandas vectorized operations.
#     """
#     # session = SessionLocal()
#     col_names = list(columns.keys())
#     rows_buffer = []  # to store each row (as a list) from the JSON Lines file
#     batch_counter = 0
#     insert_stmt = None
#     DynamicTable = None

#     # Custom loads function for orjson, since jsonlines.open() must work in text mode.
#     def custom_loads(s):
#         return orjson.loads(s.encode("utf-8"))

#     with jsonlines.open(jsonfile, mode='r', loads=custom_loads) as reader:
#         for idx, row in enumerate(reader):
#             rows_buffer.append(row)
#             if (idx + 1) % batch_size == 0:
#                 # Convert the batch to a DataFrame
#                 df = pd.DataFrame(rows_buffer, columns=col_names)
#                 # Vectorized text cleaning (apply only on object columns)
#                 for col in df.columns:
#                     if df[col].dtype == object:
#                         df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)
#                 # Convert datetime columns using vectorized pd.to_datetime.
#                 datetime_cols = [key for key in columns if columns[key] is Date]
#                 for col in datetime_cols:
#                     if col in df.columns:
#                         df[col] = pd.to_datetime(df[col], errors='coerce')
#                 # Preserve raw geometry values and initialize geometry field.
#                 if 'geocoded_column' in df.columns:
#                     df['_raw_geocoded_column'] = df['geocoded_column']
#                     df['geocoded_column'] = None

#                 # Convert the DataFrame into a list of dictionaries.
#                 batch_rows = df.to_dict(orient='records')
#                 # Convert geometry (batch-level vectorized conversion)
#                 convert_wkt(batch_rows)

#                 # On first batch, create the dynamic table and prepare the insert statement.
#                 if insert_stmt is None:
#                     DynamicTable = create_table_for_dataset(
#                         columns=dataset.col_types,
#                         prefix=dataset.short_name,
#                         engine=engine
#                     )
#                     insert_stmt = DynamicTable.__table__.insert().prefix_with("OR IGNORE")
                
#                 session.execute(insert_stmt, batch_rows)
#                 rows_buffer = []
#                 batch_counter += 1

#                 if batch_counter % commit_interval == 0:
#                     session.commit()
#                     print(f"Committed {commit_interval} batches; batch count = {batch_counter}")

#         # Process any remaining rows.
#         if rows_buffer:
#             df = pd.DataFrame(rows_buffer, columns=col_names)
#             for col in df.columns:
#                 if df[col].dtype == object:
#                     df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)
#             datetime_cols = [key for key in columns if columns[key] is Date]
#             for col in datetime_cols:
#                 if col in df.columns:
#                     df[col] = pd.to_datetime(df[col], errors='coerce')
#             if 'geocoded_column' in df.columns:
#                 df['_raw_geocoded_column'] = df['geocoded_column']
#                 df['geocoded_column'] = None
#             batch_rows = df.to_dict(orient='records')
#             convert_wkt(batch_rows)
#             session.execute(insert_stmt, batch_rows)

#         session.commit()
#     session.close()

# Explicitly define what gets imported when using `from models import *`
__all__ = [
    "table_exists",
    "write_layer_to_db",
    "create_lookup_table",
    "create_table_for_dataset",
    "populate_lookup_table",
    # "insert_dataset",
    # "prep_col_info",
    "retry",
    "set_dtypes",
    "Base",
]
