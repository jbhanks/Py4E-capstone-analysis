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
from sqlalchemy import Table
# from sqlalchemy.orm import declarative_base

import re
import time
import datetime
import pandas as pd
# import orjson
import jsonlines
from .helpers import textClean, format_float

from src.models import ColCustomization

SPECIAL_PADDING_COLS = {"tax_block" : 5, "tax_lot" : 4, "block_number": 5, "lot_number": 4, "bin" : 7 }

treat_as_string = {
    "bbl",
    "bble",
    "bin",
    "bin_number",
    "block",
    "block_number",
    "block_and_lot",
    "borough_tax_block_and_lot",
    "census",
    "census_tract",
    "council_district",
    "citycouncildistrict",
    "community_board",
    "communitydistrict",
    "communityschooldistrict",
    "congressionaldistrict",
    "designation",
    "district",
    "disposition_date",
    "e-designation",
    "fire_company",
    "geographic_area_-_2010_census_fips_county_code",
    "highousenumber",
    "house_number",
    "lot",
    "lot_number",
    "lowhousenumber",
    "number",
    "police_precinct",
    "sanitation_district_number",
    "school_district",
    "sid",
    "starfire_incident_id",
    "tax_block",
    "tax_class_code",
    "tax_lot",
    "tax_map",
    "tax_map_num",
    "tract",
    "version",
    "version_number",
    "zip",
    "zip_code",
}


class Base(DeclarativeBase):
    pass

def make_url(dbname: str, PGUSER: str, _socket: str) -> str:
    return f"postgresql+psycopg2://{PGUSER}@/{dbname}?host={_socket}"

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

from sqlalchemy import Column, Integer, String, Float, Date
from geoalchemy2 import Geometry

def create_table_for_dataset(columns, prefix, engine):
    """
    Dynamically build a SQLAlchemy ORM class for Postgres + PostGIS.
    Uses map_column_types() to resolve all column dtypes.
    """
    tabname = re.sub(r".*/", "", prefix)
    print(f"Creating table {tabname} with columns: {list(columns.keys())}")

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
                      or column in treat_as_string
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

import re
NUMERIC_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def update_numeric_columns(row, column_names, numeric_indices, new_column_types):
    """
    If a column was declared "number" but we encounter:
      - a non-numeric token (e.g., '25v2.1', 'E-61') -> String
      - a numeric with decimals                        -> Float
      - a code-like numeric (leading zeros or codey name) -> String
    """
    # CODEY_COL_TOKENS = (
    #     "zip", "zip_code",
    #     "census", "tract", "block",
    #     "district", "school_district", "police_precinct",
    #     "tax_map", "tax_map_num",
    #     "sanitation_district_number",
    #     "version", "version_number",
    #     "designation", "e-designation",  # <-- catches e-designation_number
    #     "fire_company", "tax_block", "tax_lot", "lot_number", "block_and_lot", "borough_tax_block_and_lot",
    # )

    def looks_code_like(colname: str, val_str: str) -> bool:
        if val_str.startswith("0") and len(val_str) > 1:
            return True
        name = colname.lower()
        return any(tok in name for tok in treat_as_string)

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
            print(f'Setting {colname} to String due to code-like value: {val_str}')
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


def convert_wkt(rows_to_insert):
    raw_wkts = [r.get('_raw_geocoded_column') for r in rows_to_insert]
    try:
        shapely_geoms = from_wkt(raw_wkts)
    except Exception as e:
        print(f"Error converting batch geometry: {e}")
        shapely_geoms = [None] * len(rows_to_insert)
    geoms = [from_shape(geom, srid=4326) if geom is not None else None for geom in shapely_geoms]
    for r, geom in zip(rows_to_insert, geoms):
        r['geocoded_column'] = geom
        r.pop('_raw_geocoded_column', None)

"""Reorder dictionary alphabetically, except place 'id' key first, if it exists."""
def order_columns(d):
    items = sorted((k, v) for k, v in d.items() if k != "sid")
    return dict(items[:0] + [("sid", d["sid"])] + items[0:])

def add_padding_to_special_columns(df, padding_map):
    for col, width in padding_map.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_float(x).zfill(width) if pd.notnull(x) else x)
    return df
    
def insert_dataset(engine: Engine, dataset, jsonfile, columns, batch_size=100000):
    import io
    import jsonlines
    import orjson
    import pandas as pd
    from sqlalchemy import Date  # ensure Date is in scope
    # ensure textClean is imported from your helpers at module level
    from shapely.geometry import shape as shapely_from_mapping
    from shapely import from_wkt as shapely_from_wkt

    # --- create the target table once and reuse ---
    ordered_col_types = order_columns(dataset.col_types)

    DynamicTable = create_table_for_dataset(
        columns=ordered_col_types,   # your dict of logical dtypes
        prefix=dataset.short_name,   # table name
        engine=engine
    )

    col_names = list(columns.keys())
    expected_width = len(col_names)
    rows_buffer = []
    geom_cols = []  # (col, srid)

    def custom_loads(s):
        return orjson.loads(s.encode("utf-8"))

    def normalize_rows(rows, col_names):
        out = []
        for r in rows:
            if isinstance(r, list):
                out.append({col_names[i]: (r[i] if i < len(col_names) else None) for i in range(len(col_names))})
            elif isinstance(r, dict):
                out.append(r)
            else:
                out.append(r)
        return out

    from geoalchemy2.types import Geometry as GA2Geometry  # add with other shapely imports

    def detect_geometry_columns(col_types):
        found = []
        for col, typ in col_types.items():
            # 1) Handle real Geometry types
            if isinstance(typ, GA2Geometry):
                found.append((col, getattr(typ, "srid", 4326) or 4326))
                continue
            # 2) Handle string-y declarations (e.g., "multipolygon")
            t = str(typ).lower()
            if t in ("multipolygon", "polygon", "point", "multilinestring", "linestring", "multipoint"):
                found.append((col, 4326))
        # 3) Fallback by common name if nothing detected (covers Socrata exports)
        if not found and "the_geom" in col_types:
            print("ℹ️ No declared geometry type found; treating 'the_geom' as geometry (SRID 4326).")
            found.append(("the_geom", 4326))
        return found

    def geom_to_wkt(raw):
        # Treat obvious numerics as "no geometry" to avoid parse errors
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return None
        if isinstance(raw, str):
            s = raw.strip()
            if not s or s.upper() == "NULL":
                return None
            # Numeric-looking string? bail out.
            try:
                float(s)
                return None
            except Exception:
                pass
            # Strip optional SRID prefix
            if s.upper().startswith("SRID=") and ";" in s:
                s = s.split(";", 1)[1]
            try:
                shp = shapely_from_wkt(s)
                return f"SRID=4326;{shp.wkt}"
            except Exception:
                return None
        if isinstance(raw, dict) and "type" in raw and "coordinates" in raw:
            try:
                shp = shapely_from_mapping(raw)
                return f"SRID=4326;{shp.wkt}"
            except Exception:
                return None
        return None

    _WKT_RX = re.compile(
        r'(?i)(?:SRID=\d+;)?\s*(?:MULTI(?:POINT|LINESTRING|POLYGON)|POINT|LINESTRING|POLYGON)\s*\('
        )

    def looks_like_wkt_series(s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return 0.0
        return sample.str.contains(_WKT_RX, na=False).mean()


    def looks_like_geojson_series(s: pd.Series) -> float:
        """Return ratio of values that look like GeoJSON dicts with type/coordinates."""
        vals = s.dropna().head(200).tolist()
        if not vals:
            return 0.0
        hits = 0
        for v in vals:
            if isinstance(v, dict) and "type" in v and "coordinates" in v:
                hits += 1
        return hits / len(vals)

    def is_mostly_numeric_series(s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return 0.0
        def _num(x):
            try:
                float(x)
                return True
            except Exception:
                return False
        return sample.apply(_num).mean()

    def detect_json_columns(col_types):
        cols = []
        for col, typ in col_types.items():
            # SQLAlchemy types
            if isinstance(typ, (JSON, JSONB)):
                cols.append(col)
                continue
            # String-y declarations like "json"/"jsonb"
            t = str(typ).lower()
            if t in ("json", "jsonb"):
                cols.append(col)
        return cols

    def _coerce_socrata_location(value):
        """
        Socrata sometimes provides location as:
        [human_address_json_string, latitude, longitude, <unused>, needs_recoding_bool]
        Convert to a single dict.
        """
        if not isinstance(value, list):
            return value
        if len(value) >= 3:
            human = value[0]
            # human_address may itself be a JSON string
            if isinstance(human, str):
                try:
                    human = orjson.loads(human)
                except Exception:
                    pass
            out = {
                "human_address": human,
                "latitude": value[1],
                "longitude": value[2],
            }
            if len(value) >= 5:
                out["needs_recoding"] = bool(value[4])
            return out
        return value

    # add with your imports if not already present
    import ast
    from sqlalchemy.dialects.postgresql import JSON, JSONB  # you already used these in detect_json_columns

    def detect_json_like_columns_in_df(df: pd.DataFrame, declared_json_cols: list[str]) -> list[str]:
        """
        Return columns that either:
        - were declared JSON/JSONB, or
        - contain dict/list objects, or
        - are strings that look like JSON/Python-lists that we can parse.
        """
        cols = set(declared_json_cols)

        # Heuristic: common Socrata name
        if "location" in df.columns:
            cols.add("location")

        sample_n = 100
        for c in df.columns:
            s = df[c].dropna().head(sample_n)
            if s.empty:
                continue
            # already Python dict/list?
            if any(isinstance(v, (dict, list)) for v in s):
                cols.add(c)
                continue
            # string that looks like JSON / Python list?
            looks_like = s.astype(str).str.strip().str.startswith(("{", "[")).mean() > 0.2
            if looks_like:
                # prove we can parse at least one value
                ok = False
                for v in s.astype(str):
                    t = v.strip()
                    try:
                        orjson.loads(t)
                        ok = True
                        break
                    except Exception:
                        try:
                            _ = ast.literal_eval(t)
                            ok = True
                            break
                        except Exception:
                            pass
                if ok:
                    cols.add(c)
        return list(cols)


    def normalize_json_value(v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        py = v
        # Parse strings into Python values
        if isinstance(v, str):
            s = v.strip()
            try:
                py = orjson.loads(s)
            except Exception:
                try:
                    # Handle Python reprs like "['...', None, False]"
                    py = ast.literal_eval(s)
                except Exception:
                    # Leave as plain string
                    py = s
        # Fix Socrata location list -> dict
        py = _coerce_socrata_location(py)
        # Dump to proper JSON text (double quotes, true/false/null)
        try:
            return orjson.dumps(py).decode("utf-8")
        except Exception:
            return None


    def ensure_geometry_columns(df: pd.DataFrame, geom_cols: list[tuple[str,int]]) -> pd.DataFrame:
        """
        Guarantee that each declared geometry column contains EWKT or NULL.
        If a geometry column is numeric/malformed, try to source geometry from
        another column that looks like WKT/GeoJSON. Convert to EWKT via geom_to_wkt.
        """
        # 1) Try to fix obvious misalignment by copying from the best-looking geometry column
        candidates = []
        for c in df.columns:
            wkt_ratio = looks_like_wkt_series(df[c])
            gj_ratio  = looks_like_geojson_series(df[c])
            if (wkt_ratio >= 0.2) or (gj_ratio >= 0.2):
                candidates.append((c, wkt_ratio, gj_ratio))
        # Rank candidates by (geojson or wkt ratio), favor WKT a little
        candidates.sort(key=lambda x: (max(x[1], x[2]), x[1]), reverse=True)

        for gcol, _srid in geom_cols:
            if gcol not in df.columns:
                # If missing, create and fill from best candidate if any
                src = candidates[0][0] if candidates else None
                df[gcol] = df[src] if src else None
                if src:
                    print(f"🔧 geometry column '{gcol}' missing — copying from '{src}'")
                continue

            num_ratio = is_mostly_numeric_series(df[gcol])
            wkt_ratio = looks_like_wkt_series(df[gcol])
            gj_ratio  = looks_like_geojson_series(df[gcol])

            if num_ratio > 0.7 and max(wkt_ratio, gj_ratio) < 0.2:
                # very likely mis-mapped; try best candidate
                if candidates:
                    src = candidates[0][0]
                    if src != gcol:
                        print(f"🔧 geometry misalignment: mapping '{src}' → '{gcol}'")
                        df[gcol] = df[src]

            # Finally: convert whatever is in gcol to EWKT (or None)
            df[gcol] = df[gcol].apply(geom_to_wkt)

            # Fail-fast guard: no numeric leftovers
            after_num_ratio = is_mostly_numeric_series(df[gcol])
            if after_num_ratio > 0.0:
                # Nuke numeric stragglers to NULL so COPY can't choke
                def _clean_num_to_none(v):
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        return None
                    if isinstance(v, str):
                        try:
                            float(v.strip())
                            return None
                        except Exception:
                            return v
                    return v
                df[gcol] = df[gcol].apply(_clean_num_to_none)

        return df


    def relink_geometry_sources(df: pd.DataFrame, geom_cols: list[str]) -> pd.DataFrame:
        """If a declared geometry column is numeric/malformed, try to find the real geometry column in df."""
        for gcol, _srid in geom_cols:
            if gcol not in df.columns:
                continue
            col = df[gcol]
            # If this geometry column is numeric for most rows, it's likely mis-mapped
            numeric_ratio = pd.to_numeric(col, errors="coerce").notna().mean() if len(col) else 0.0
            looks_like_wkt = looks_like_wkt_series(col)
            looks_like_geojson = looks_like_geojson_series(col)

            if numeric_ratio > 0.7 and not looks_like_wkt and not looks_like_geojson:
                # try to discover a better source
                candidates = []
                for c in df.columns:
                    if c == gcol:
                        continue
                    if looks_like_wkt_series(df[c]) or looks_like_geojson_series(df[c]):
                        candidates.append(c)
                if candidates:
                    src = candidates[0]
                    print(f"🔧 Detected geometry misalignment: mapping '{src}' → '{gcol}'")
                    df[gcol] = df[src]
        return df

    def process_batch(df, geom_cols, json_cols):
        df = df.where(pd.notnull(df), None)

        # Sanitize strings
        NUMERICISH_COLS = {"latitude", "longitude", "lat", "lon", "lng", "x", "y", "issue_date", "x_coord_cd", "y_coord_cd", "x_coord", "y_coord"}

        for col in df.columns:
            if col in treat_as_string:
                df[col] = df[col].apply(lambda x: format_float(x) if pd.notnull(x) else x).astype("string")
            elif df[col].dtype == object:
                # If this column is mostly numeric, convert to numeric and skip textClean
                numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
                if col.lower() in NUMERICISH_COLS or numeric_ratio >= 0.8:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].apply(lambda x: textClean(x) if isinstance(x, str) else x)

        # Dates → date objects
        datetime_cols = [key for key, typ in columns.items() if typ is Date]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].apply(lambda x: x.date() if pd.notnull(x) else None)

        # 🔎 Try to fix obvious geometry mis-mapping before conversion
        df = relink_geometry_sources(df, geom_cols)

        # Geometry → EWKT text (explicit SRID)
        for col, _srid in geom_cols:
            if col in df.columns:
                df[col] = df[col].apply(geom_to_wkt)
          
        # Normalize JSON columns to valid JSON text
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalize_json_value)

        json_like_cols = detect_json_like_columns_in_df(df, json_cols)
        for col in json_like_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalize_json_value)

        return df

    def verify_batch(engine, table_like, sid_series, sid_col="sid", sample_missing=10):
        import io
        import pandas as pd

        table = getattr(table_like, "__table__", table_like)
        tgt_schema = getattr(table, "schema", None)
        tgt_table = f'"{tgt_schema}"."{table.name}"' if tgt_schema else f'"{table.name}"'

        # Build a one-column CSV buffer for fast COPY
        buf = io.StringIO()
        pd.Series(sid_series, name="sid").to_csv(buf, index=False, header=False)
        buf.seek(0)

        tmp = "tmp_expected_sids"

        conn = engine.raw_connection()
        try:
            cur = conn.cursor()

            # NOTE: no "ON COMMIT DROP"
            cur.execute(f'CREATE TEMP TABLE {tmp} (sid text);')

            # Copy the SIDs into the temp table (same transaction)
            cur.copy_expert(f"COPY {tmp} (sid) FROM STDIN WITH (FORMAT CSV)", buf)

            # Now run the comparisons BEFORE any commit
            cur.execute(f"SELECT COUNT(*) FROM {tmp};")
            expected = cur.fetchone()[0]

            cur.execute(f'SELECT COUNT(*) FROM {tgt_table} t JOIN {tmp} x ON t."{sid_col}" = x.sid;')
            found = cur.fetchone()[0]

            missing = []
            if found != expected:
                cur.execute(
                    f'''
                    SELECT x.sid
                    FROM {tmp} x
                    LEFT JOIN {tgt_table} t ON t."{sid_col}" = x.sid
                    WHERE t."{sid_col}" IS NULL
                    LIMIT %s;
                    ''',
                    (sample_missing,)
                )
                missing = [r[0] for r in cur.fetchall()]

            print(f"🔍 Batch verify: expected={expected}, found={found}, missing={expected - found}")
            if missing:
                print("   e.g. missing SIDs:", missing)

            # Clean up and commit at the end
            cur.execute(f"DROP TABLE IF EXISTS {tmp};")
            conn.commit()
            return expected, found, missing

        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()


    def enforce_integer_compat(df, table_class):
        """Coerce whole-number floats/strings to Int64 for integer/bigint columns.
        If fractional values exist in an integer column, raise with examples.
        """
        int_cols = [
            c.name for c in table_class.__table__.columns
            if isinstance(getattr(c, "type", None), (Integer, BigInteger)) and c.name != "id"
        ]
        for c in int_cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")

            # detect true fractional values
            frac_mask = s.notna() & ((s % 1) != 0)
            if frac_mask.any():
                examples = s[frac_mask].head(5).tolist()
                raise ValueError(
                    f"Column '{c}' is INTEGER/BIGINT but has fractional values, e.g. {examples}. "
                    f"Either clean your data or change the column to NUMERIC(14,2) (recommended for money-like fields)."
                )

            # coerce whole numbers (including '85392.00') to nullable int
            df[c] = s.astype("Int64")
        return df

    def copy_batch(engine, df, table_like):
        """
        COPY df -> table_like. Accepts either a Declarative class (with __table__)
        or a plain sqlalchemy.Table.
        """
        import io

        # Resolve to a sqlalchemy.Table
        table = getattr(table_like, "__table__", table_like)
        if not hasattr(table, "name"):
            raise TypeError("copy_batch expected a Declarative class or sqlalchemy.Table")

        # Build fully-qualified table name
        name = table.name
        schema = getattr(table, "schema", None)
        if schema:
            fqtn = f'"{schema}"."{name}"'
        else:
            fqtn = f'"{name}"'

        # Columns to COPY (skip autoincrement PKs like 'id' if present)
        copy_cols = [c.name for c in table.columns if c.name != "id"]

        # Ensure df has all COPY columns and in correct order
        for col in copy_cols:
            if col not in df.columns:
                df[col] = None
        df = df[copy_cols]

        # Prepare CSV buffer (NULL -> \N for COPY)
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)

        # Execute COPY
        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            collist_sql = ", ".join(f'"{c}"' for c in copy_cols)
            sql = f"COPY {fqtn} ({collist_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
            cur.copy_expert(sql, buf)
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

    # Detect geometry columns once
    geom_cols = detect_geometry_columns(dataset.col_types)
    # Detect JSON columns once
    json_cols = detect_json_columns(dataset.col_types)


    # --- Main loop ---

    with jsonlines.open(jsonfile, mode="r", loads=custom_loads) as reader:
        batch_start = time.perf_counter()
        for idx, row in enumerate(reader):
            if isinstance(row, list) and len(row) > expected_width:
                row = row[:expected_width]
            rows_buffer.append(row)

            if (idx + 1) % batch_size == 0:
                normed = normalize_rows(rows_buffer, col_names)
                df = pd.DataFrame(normed)
                # print(f"First df.head is {df.head().to_dict(orient='records')}")
                if not geom_cols and "the_geom" in df.columns:
                    geom_cols = [("the_geom", 4326)]
                df = process_batch(df, geom_cols, json_cols)
                df = ensure_geometry_columns(df, geom_cols)
                df = enforce_integer_compat(df, DynamicTable)
                df = add_padding_to_special_columns(df, SPECIAL_PADDING_COLS)
                copy_batch(engine, df, DynamicTable)
                # Verify this batch by SID
                if "sid" in df.columns:
                    _exp, _found, _missing = verify_batch(engine, DynamicTable, df["sid"])
                else:
                    print("⚠️ No 'sid' column available to verify this batch.")
                t3 = time.perf_counter()
                print(f"📤 Inserted {len(df)} rows in {t3 - batch_start:.2f}s")
                rows_buffer.clear()
                batch_start = time.perf_counter()

        # Final flush
        if rows_buffer:
            normed = normalize_rows(rows_buffer, col_names)
            df = pd.DataFrame(normed)
            if not geom_cols and "the_geom" in df.columns:
                geom_cols = [("the_geom", 4326)]
            df = process_batch(df, geom_cols, json_cols)
            df = ensure_geometry_columns(df, geom_cols)
            df = enforce_integer_compat(df, DynamicTable)
            df = add_padding_to_special_columns(df, SPECIAL_PADDING_COLS)
            # Optional: quick peek to prove it's not numeric anymore
            copy_batch(engine, df, DynamicTable)
            if "sid" in df.columns:
                _exp, _found, _missing = verify_batch(engine, DynamicTable, df["sid"])
            else:
                print("⚠️ No 'sid' column available to verify this batch.")
            print(f"✅ Final batch inserted ({len(df)} rows)")

    # Final sanity check: count source vs target rows
    # Count source rows (stream)
    src_count = 0
    with jsonlines.open(dataset.dataset_path, mode="r", loads=lambda s: orjson.loads(s.encode("utf-8"))) as rdr:
        for _ in rdr:
            src_count += 1

    # Count target rows
    from sqlalchemy import text
    with engine.connect() as conn:
        tgt_count = conn.execute(text(f'SELECT COUNT(*) FROM "{DynamicTable.__table__.name}"')).scalar_one()

    print(f"Source rows: {src_count}  |  Target rows: {tgt_count}")


# Explicitly define what gets imported when using `from models import *`
__all__ = [
    "table_exists",
    "write_layer_to_db",
    "create_lookup_table",
    "create_table_for_dataset",
    "populate_lookup_table",
    "insert_dataset",
    # "prep_col_info",
    "make_url",
    "retry",
    "treat_as_string",
    "set_dtypes",
    "SPECIAL_PADDING_COLS",
    "add_padding_to_special_columns",
    "Base",
]
