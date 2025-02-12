from geoalchemy2 import Geometry
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, Date, LargeBinary, TIMESTAMP, MetaData, Table, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert
import geopandas as gpd

# from sqlalchemy.sql import text




# Define the base class
Base = declarative_base()

# Function to map pandas dtype to SQLAlchemy Column type
def map_dtype(dtype):
    if isinstance(dtype, gpd.array.GeometryDtype):
        return LargeBinary
    elif np.issubdtype(dtype, np.integer):
        return Integer
    elif np.issubdtype(dtype, np.floating):
        return Float
    elif np.issubdtype(dtype, np.bool_):
        return Boolean
    elif dtype == 'object':
        return String
    elif np.issubdtype(dtype, np.datetime64):
        return TIMESTAMP
    elif np.issubdtype(dtype, np.timedelta64):
        return String
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# Function to create mixin class with unshared columns
def create_mixin_class(table_name, unshared_columns, df, dtype_exceptions):
    class Mixin(Base):
        __abstract__ = True

    for col in unshared_columns:
        dtype = dtype_exceptions.get(col, map_dtype(df[col].dtype))
        if isinstance(dtype, type):
            setattr(Mixin, col, Column(dtype))
        else:
            raise TypeError(f"Invalid dtype found in dtype_exceptions for '{col}': {dtype}")

    Mixin.__name__ = f"{table_name}Mixin"
    return Mixin

# Function to create ORM classes
def create_orm_classes(df_dict, base_class, dtype_exceptions):
    orm_classes = {}
    base_class.metadata.clear()
    for name, df in df_dict.items():
        table_name = name.lower()
        class_name = name.capitalize()

        attrs = {
            '__tablename__': table_name,
            'id': Column(Integer, primary_key=True, autoincrement=True),
        }

        shared_columns = {}
        for col in df.columns:
            try:
                column_type = dtype_exceptions.get(col, map_dtype(df[col].dtype))
                if not isinstance(column_type, type) or not issubclass(column_type, sqlalchemy.types.TypeEngine):
                    raise TypeError(f"Invalid SQLAlchemy type for column '{col}': {column_type}")
                shared_columns[col] = Column(column_type)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue

        attrs.update(shared_columns)
        unshared_columns = [col for col in df.columns if col not in shared_columns]
        mixin = create_mixin_class(name, unshared_columns, df, dtype_exceptions) if unshared_columns else base_class
        orm_classes[class_name] = type(class_name, (mixin,), attrs)

    return orm_classes

