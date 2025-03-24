from sqlalchemy import Engine
from sqlalchemy import (
    inspect,
    Column,
    Integer,
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

import re
import datetime
import pandas as pd
import orjson
import jsonlines
from .helpers import merge_dicts, parseDateString, split_address, textClean


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
    args = {
        "primary_key": False,  # Marks the column as a primary key.
        "nullable": True,  # Disallows NULL values.
        "unique": False,  # Enforces unqiue values.
        "index": True,  # Creates an index on this column.
        "default" : None, # Python-side default
        "server_default": 'NULL',  # SQL-side default value (as a string).
        "autoincrement": True,  # Controls autoincrement behavior.
    }
    return Column(colname, dtype, **args)

def create_table(columns, engine, tabname):
    columns = {'local_id': Column(Integer, primary_key=True)} | {colname: set_column(colname, dtype) for colname,dtype in columns.items() if isinstance(colname, str)}
    columns['sid'] = Column(String, unique=True, nullable=False)
    table_class = type(f"{tabname.capitalize()}Table", (Base,), {
        "__tablename__": f"{tabname}",
        "__table_args__": {'extend_existing': True},
        **columns
    })
    print(f'Creating table class {tabname}: {table_class}')
    # Create the table in the database
    Base.metadata.create_all(engine)
    return table_class

def create_table_for_dataset(columns, prefix, engine):
    tabname = re.sub('.*/', '', prefix)
    print(f'Creating table {tabname}')
    DynamicTable = create_table(columns, engine, tabname)
    if DynamicTable is None:
        raise(Exception("DynamicTable not made!"))
    return DynamicTable

# def create_table(column_names, data_list, engine, tabname):
#     columns = {
#         "local_id": Column(Integer, primary_key=True)
#     }  # Adding a primary key column
#     for col, data in tuple(zip(column_names, data_list)):
#         if col[1] is String:
#             columns[col[0]] = Column(col[1], default="NO DATA")
#         elif col[1] is Integer:
#             columns[col[0]] = Column(col[1], default=0)
#         elif col[1] is Date:
#             columns[col[0]] = Column(col[1])
#     # Dynamically create a unique ORM class for the table
#     table_class = type(
#         f"{tabname.capitalize()}Table",
#         (Base,),
#         {
#             "__tablename__": f"{tabname}",
#             "__table_args__": {"extend_existing": True},
#             **columns,
#         },
#     )
#     # Create the table in the database
#     Base.metadata.create_all(engine)
#     return table_class


# def create_table_for_dataset(keepcols, prefix, row, engine):
#     cols = keepcols
#     tabname = re.sub(".*/", "", prefix)
#     DynamicTable = create_table(cols, row, engine, tabname)
#     if DynamicTable is None:
#         raise (Exception("DynamicTable not made!"))
#     return DynamicTable



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

# def insert_dataset(
#     session, jsonfile, engine, prefix, keepcols, column_names, datetime_cols
# ):
#     with open(jsonfile, "rb") as f:
#         batch_size = 1000
#         rows_to_insert = []
#         for idx, row in enumerate(ijson.items(f, "item")):
#             if idx == 0:
#                 DynamicTable = create_table_for_dataset(
#                     column_names, keepcols, prefix, row, engine
#                 )
#             row = [textClean(val) if type(val) is str else val for val in row]
#             row_data = dict(
#                 zip(
#                     [col[0] for col in keepcols],
#                     [row[column_names.index(col[0])] for col in keepcols],
#                 )
#             )
#             row_data = {
#                 key: (
#                     parseDateString(row_data[key])
#                     if key in datetime_cols and row_data[key] is not None
#                     else row_data[key]
#                 )
#                 for key in row_data.keys()
#             }
#             # This is to deal with a peculiarity in of the datasets
#             row_data = split_address(row_data)
#             rows_to_insert.append(row_data)
#             if len(rows_to_insert) == 0:
#                 raise (Exception("Row empty!"))
#             if (idx + 1) % batch_size == 0:
#                 session.bulk_insert_mappings(DynamicTable, rows_to_insert)
#                 session.commit()
#                 rows_to_insert = []
#         if rows_to_insert:
#             try:
#                 session.bulk_insert_mappings(DynamicTable, rows_to_insert)
#                 session.commit()
#             except Exception as e:
#                 print(f"Error inserting batch at index {idx}: {e}")
#                 session.rollback()
#     session.close()


# def prep_col_info(configs, name):
#     prefix = f"{configs['prefix']}/{name}"
#     cols_to_drop = configs["cols_to_drop"]
#     cols_to_rename = configs["cols_to_rename"]
#     datype_exceptions = configs["datype_exceptions"]
#     datatype_mappings = configs["datatype_mappings"]
#     with open(f"{prefix}_data_types.txt", "r") as f:
#         data_types = f.read().replace('"', "").strip().lower().split("\n")
#         for idx, datatype in enumerate(data_types):
#             if datatype in datatype_mappings.keys():
#                 data_types[idx] = datatype_mappings[datatype]
#     with open(prefix + "_colnames.txt", "r") as f:
#         # Clean up the column names
#         column_names = f.read().replace('"', "").split("\n")
#         column_names = fix_colnames(column_names, cols_to_rename)
#     cols = dict(zip(column_names, data_types))
#     for key in cols.keys():
#         if key in datype_exceptions.keys():
#             cols[key] = datype_exceptions[key]
#     # Drop columns that I don't think I will need
#     keepcols = [(col, cols[col]) for col in cols.keys() if col not in cols_to_drop]
#     datetime_cols = [
#         col[0] if col[1] is Date else None for idx, col in enumerate(keepcols)
#     ]
#     datetime_cols = [col for col in datetime_cols if col is not None]
#     return column_names, cols, keepcols, datetime_cols


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

def map_column_types(column_types):
    """Map column names to SQLAlchemy types based on the given column types."""
    return {
        column: (
            Integer if dtype == "number" or column in ['created_at', 'updated_at'] else
            String if dtype == "text" or column == "sid" else
            Date if dtype == "calendar_date" else
            JSON if dtype == "location" else
            Geometry(geometry_type='POINT', srid=4326) if dtype == "point" else
            Geometry(geometry_type='MULTIPOLYGON', srid=4326) if dtype == "multipolygon" else
            (Geometry(geometry_type='POLYGON', srid=4326) if dtype == "polygon" else dtype)
        )
        for column, dtype in column_types.items() if not column.startswith(':@')
    }

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

def update_numeric_columns(row, column_names, column_types, numeric_indices, new_column_types):
    """Check numeric columns and update the column types accordingly."""
    row_list = [row.get(col) for col in column_names]
    for idx in numeric_indices:
        value = row_list[idx]
        if value is None:
            continue  # Skip null values
        if isinstance(value, (int, float)):
            if float(value) % 1 != 0:
                colname = column_names[idx]
                new_column_types[colname] = Float
                break
        elif isinstance(value, str):
            val_str = value.strip()
            if not val_str.isdigit() and is_number(val_str):
                if float(val_str) % 1 != 0:
                    colname = column_names[idx]
                    new_column_types[colname] = Float
                    break
    return new_column_types

def set_dtypes(filename, column_types):
    """Main function to process the file and set data types."""
    print(f'Starting {filename}')
    if not filename.endswith('.json'):
        print(f'{filename} is not a json file, skipping for now...')
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
            new_column_types = update_numeric_columns(row, column_names, column_types, numeric_indices, new_column_types)

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

def insert_dataset(dataset, jsonfile, columns, batch_size=10000, commit_interval=100):
    """
    commit_interval: How many batches to accumulate before a commit.
    """
    session = SessionLocal()
    col_names = list(columns.keys())
    rows_to_insert = []
    batch_counter = 0
    
    # Define a custom loads function for orjson.
    def custom_loads(s):
        return orjson.loads(s.encode("utf-8"))
    
    with jsonlines.open(jsonfile, mode='r', loads=custom_loads) as reader:
        for idx, row in enumerate(reader):
            if idx == 0:
                datetime_cols = [key for key in columns if columns[key] is Date]
                DynamicTable = create_table_for_dataset(
                    columns=dataset.col_types, 
                    prefix=dataset.short_name, 
                    engine=engine
                )
                # Prepare the insert statement (using SQLite's OR IGNORE to skip duplicates)
                insert_stmt = DynamicTable.__table__.insert().prefix_with("OR IGNORE")
            
            # Row is assumed to be a list; clean text values.
            row = [textClean(val) if isinstance(val, str) else val for val in row]
            # Map list to dict using the expected order.
            row_data = {col_name: (row[i] if i < len(row) else None)
                        for i, col_name in enumerate(col_names)}
            
            # Keep the raw WKT for later geometry conversion.
            row_data['_raw_geocoded_column'] = row_data.get('geocoded_column')
            row_data['geocoded_column'] = None  # initialize to None for now.
            
            # Process datetime values.
            temp = {}
            for key in row_data:
                if key == '_raw_geocoded_column':
                    temp[key] = row_data[key]
                else:
                    temp[key] = (parseDateString(row_data[key])
                                 if key in datetime_cols and row_data[key] is not None
                                 else row_data[key])
            row_data = temp
            
            rows_to_insert.append(row_data)
            
            if (idx + 1) % batch_size == 0:
                convert_wkt(rows_to_insert)
                # Batch convert geometries.
                session.execute(insert_stmt, rows_to_insert)
                rows_to_insert = []
                batch_counter += 1
                
                # Rather than commit for each batch, commit once every commit_interval batches.
                if batch_counter % commit_interval == 0:
                    commit_start = time.perf_counter()
                    session.commit()
                    commit_end = time.perf_counter()
                    print(f"Committed {commit_interval} batches in {commit_end - commit_start:.3f} seconds.")
                    # Optionally, you can also print the batch counter.
                    print(f"Batch {batch_counter} processed.")

        # Process any leftover rows.
        if rows_to_insert:
            convert_wkt(rows_to_insert)
            session.execute(insert_stmt, rows_to_insert)
        
        # Final commit for any outstanding operations.
        session.commit()
    session.close()

# Explicitly define what gets imported when using `from models import *`
__all__ = [
    "table_exists",
    "write_layer_to_db",
    "create_lookup_table",
    "create_table_for_dataset",
    "populate_lookup_table",
    "insert_dataset",
    # "prep_col_info",
    "retry",
    "set_dtypes",
    "Base",
]
