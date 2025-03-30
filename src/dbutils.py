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

def map_column_types(column_types):
    """Map column names to SQLAlchemy types based on the given column types."""
    return {
        column: (
            Integer if dtype == "number" or column in ['created_at', 'updated_at'] or column.startswith('int') else
            String if dtype == "text" or column == "sid" or column.startswith('str') else
            Date if dtype == "calendar_date" else
            Float if dtype.startswith('float') else
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

def update_numeric_columns(row, column_names, numeric_indices, new_column_types):
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


def insert_dataset(engine, dataset, jsonfile, columns, batch_size=10000, commit_interval=100):
    """
    Process the JSON Lines file in batches using pandas vectorized operations.
    """
    session = SessionLocal()
    col_names = list(columns.keys())
    rows_buffer = []  # to store each row (as a list) from the JSON Lines file
    batch_counter = 0
    insert_stmt = None
    DynamicTable = None

    # Custom loads function for orjson, since jsonlines.open() must work in text mode.
    def custom_loads(s):
        return orjson.loads(s.encode("utf-8"))

    with jsonlines.open(jsonfile, mode='r', loads=custom_loads) as reader:
        for idx, row in enumerate(reader):
            rows_buffer.append(row)
            if (idx + 1) % batch_size == 0:
                # Convert the batch to a DataFrame
                df = pd.DataFrame(rows_buffer, columns=col_names)
                # Vectorized text cleaning (apply only on object columns)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)
                # Convert datetime columns using vectorized pd.to_datetime.
                datetime_cols = [key for key in columns if columns[key] is Date]
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                # Preserve raw geometry values and initialize geometry field.
                if 'geocoded_column' in df.columns:
                    df['_raw_geocoded_column'] = df['geocoded_column']
                    df['geocoded_column'] = None

                # Convert the DataFrame into a list of dictionaries.
                batch_rows = df.to_dict(orient='records')
                # Convert geometry (batch-level vectorized conversion)
                convert_wkt(batch_rows)

                # On first batch, create the dynamic table and prepare the insert statement.
                if insert_stmt is None:
                    DynamicTable = create_table_for_dataset(
                        columns=dataset.col_types,
                        prefix=dataset.short_name,
                        engine=engine
                    )
                    insert_stmt = DynamicTable.__table__.insert().prefix_with("OR IGNORE")
                
                session.execute(insert_stmt, batch_rows)
                rows_buffer = []
                batch_counter += 1

                if batch_counter % commit_interval == 0:
                    session.commit()
                    print(f"Committed {commit_interval} batches; batch count = {batch_counter}")

        # Process any remaining rows.
        if rows_buffer:
            df = pd.DataFrame(rows_buffer, columns=col_names)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)
            datetime_cols = [key for key in columns if columns[key] is Date]
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            if 'geocoded_column' in df.columns:
                df['_raw_geocoded_column'] = df['geocoded_column']
                df['geocoded_column'] = None
            batch_rows = df.to_dict(orient='records')
            convert_wkt(batch_rows)
            session.execute(insert_stmt, batch_rows)

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
