from sqlalchemy import Engine
from sqlalchemy import inspect, create_engine, Column, Integer, String, Date, MetaData, event, Table, text, LargeBinary

import pandas as pd
import datetime
from src.helpers import merge_dicts
# from sqlalchemy.orm import sessionmaker
# import geopandas as gpd



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
        batch = gdf.iloc[start:start + batch_size]
        for _, row in batch.iterrows():
            if 'geometry' in gdf.columns:
                geometry_wkb = row['geometry'].wkb if row['geometry'] else None
            
            # Convert date fields to Python date objects
            for date_field in ['YearBuilt', 'YearAlter1', 'YearAlter2', 'APPDate']:
                if pd.notnull(row[date_field]) and row[date_field] != 0:
                    try:
                        # Handle year-only format
                        row[date_field] = datetime.datetime.strptime(str(int(row[date_field])), '%Y').date()
                    except ValueError:
                        try:
                            # Handle full date format
                            row[date_field] = datetime.datetime.strptime(str(row[date_field]), '%m/%d/%Y').date()
                        except ValueError:
                            row[date_field] = None
                else:
                    row[date_field] = None

            if 'geometry' in gdf.columns:
                entry = orm_classes[layer_name.capitalize()](
                    geometry=geometry_wkb,
                    **{col: row[col] for col in gdf.columns if col not in ['geometry']}
                )
            else:
                entry = orm_classes[layer_name.capitalize()](
                    **{col: row[col] for col in gdf.columns if col not in ['geometry']}
                )
            session.add(entry)
        session.commit()


def create_lookup_table(engine, lookup_table_name, text_column_name):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    if table_exists(engine, lookup_table_name):
        print("Table exists")
    else:
        # Step 3: Create the lookup table with ID and category columns
        lookup_table = Table(
            lookup_table_name,
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column(text_column_name, String, unique=True, nullable=False, default="NO DATA"),
            Column('info', String, unique=False, nullable=True, default=None)
        )
        lookup_table.create(engine)
        return lookup_table


# def create_lookup_table(engine, lookup_table_name, columns):
#     """
#     Create a lookup table with specified columns.

#     Args:
#         engine (Engine): The SQLAlchemy engine connected to the database.
#         lookup_table_name (str): The name of the lookup table to create.
#         columns (list of tuples): A list of tuples where each tuple contains the column name and its creation arguments.

#     Returns:
#         Table: The created lookup table.
#     """
#     metadata = MetaData()
#     metadata.reflect(bind=engine)
#     if table_exists(engine, lookup_table_name):
#         print("Table exists")
#     else:
#         # Step 3: Create the lookup table with specified columns
#         lookup_table = Table(
#             lookup_table_name,
#             metadata,
#             Column('id', Integer, primary_key=True, autoincrement=True),
#             *[Column(col_name, *col_args) for col_name, *col_args in columns]
#         )
#         lookup_table.create(engine)
#         return lookup_table

def create_table_for_dataset(column_names, keepcols, prefix, row, engine):
    if 'staddr' in column_names:
        cols = keepcols + [('building_num', Integer), ('street_name', String)]
    else:
        cols = keepcols
    tabname = re.sub('.*/', '', prefix)
    DynamicTable = create_table(cols, row, engine, tabname)
    if DynamicTable is None:
        raise(Exception("DynamicTable not made!"))
    return DynamicTable

def insert_dataset(session, filename, engine, prefix, keepcols, column_names, datetime_cols):
    with open(filename, "rb") as f:
        batch_size = 1000
        rows_to_insert = []
        for idx, row in enumerate(ijson.items(f, "item")):
            if idx == 0:
                DynamicTable = create_table_for_dataset(column_names, keepcols, prefix, row, engine)
            row = [textClean(val) if type(val) is str else val for val in row ]
            row_data = dict(zip([col[0] for col in keepcols], [row[column_names.index(col[0])] for col in keepcols]))
            row_data = { key:(parseDateString(row_data[key]) if key in datetime_cols and row_data[key] is not None else row_data[key]) for key in row_data.keys()}
            # This is to deal with a peculiarity in of the datasets
            row_data = split_address(row_data)
            rows_to_insert.append(row_data)
            if len(rows_to_insert) == 0:
                raise(Exception("Row empty!"))
            if (idx + 1) % batch_size == 0:
                session.bulk_insert_mappings(DynamicTable, rows_to_insert)
                session.commit()
                rows_to_insert = []
        if rows_to_insert:
            try:
                session.bulk_insert_mappings(DynamicTable, rows_to_insert)
                session.commit()
            except Exception as e:
                print(f"Error inserting batch at index {idx}: {e}")
                session.rollback()
    session.close()


def prep_col_info(configs, name):
    prefix = f"{configs['prefix']}/{name}"
    cols_to_drop = configs['cols_to_drop']
    cols_to_rename = configs['cols_to_rename']
    datype_exceptions = configs['datype_exceptions']
    datatype_mappings = configs['datatype_mappings']
    with open(f'{prefix}_data_types.txt', 'r') as f:
        data_types = f.read().replace('"', '').strip().lower().split('\n')
        for idx,datatype in enumerate(data_types):
            if datatype in datatype_mappings.keys():
                data_types[idx] = datatype_mappings[datatype]
    with open(prefix + '_colnames.txt', 'r') as f:
        # Clean up the column names
        column_names = f.read().replace('"', '').split('\n')
        column_names = fix_colnames(column_names, cols_to_rename)
    cols = dict(zip(column_names, data_types))
    for key in cols.keys():
        if key in datype_exceptions.keys():
            cols[key] = datype_exceptions[key]
    # Drop columns that I don't think I will need
    keepcols = [(col, cols[col]) for col in cols.keys() if col not in cols_to_drop]
    # print("keepcols is", keepcols)
    datetime_cols = [col[0] if col[1] is Date else None for idx,col in enumerate(keepcols)]
    datetime_cols = [col for col in datetime_cols if col is not None]
    return column_names, cols, keepcols, datetime_cols

def preprocess_dataset(engine, shared_dataset_configs, specific_dataset_config, name):
    configs = merge_dicts(shared_dataset_configs, specific_dataset_config) 
    column_names, cols, keepcols, datetime_cols = prep_col_info(configs, name)
    prefix = configs['prefix'] + '/' + name
    insert_dataset(prefix + "_rows.json", engine,prefix, keepcols, column_names, datetime_cols)


# Explicitly define what gets imported when using `from models import *`
__all__ = ['table_exists', 'write_layer_to_db', 'create_lookup_table', 'create_table_for_dataset', 'insert_dataset', 'prep_col_info']