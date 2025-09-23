
# # Design the database based on the data
# * #### This notebook parses metadata associated with some of the datasets, most especially the PLUTO dataset, which contains columns that are also in many other datasets I looked at on NYCOpenData.
# * #### In some cases I had to search around to find more complete definitions than were included in the data dictionary associated with the dataset.


import pdfplumber
import pandas as pd
import time
import dill
time.sleep(5) # This seems to be necessary to avoid an error about dill not being loaded in the next cell. Sometimes even that is not enough and this cell needs to be run again.
from bisect import bisect_left
from itertools import tee
from src.models import ColCustomization
from src.helpers import *
from src.pdfutils import *
from src.dbutils import *
from geoalchemy2 import Geometry
from sqlalchemy import String


# * Load the objects created in the previous notebook


# Load the environment
with open("environment_data/select.pkl", "rb") as f:
    env = dill.load(f)

# Restore the environment
globals().update(env)


dataset_info_dict


if 'assessments' in dataset_info_dict:
    column_types = dataset_info_dict['assessments'].col_types
    print(column_types)
    print(len(column_types))

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
#                 break
#         elif isinstance(value, str):
#             print(f'Verifying value: {value} (type: {type(value)})')
#             val_str = value.strip()
#             if not val_str.isdigit() and is_number(val_str):
#                 if float(val_str) % 1 != 0:
#                     colname = column_names[idx]
#                     new_column_types[colname] = Float
#                     break
#         else:
#             print(f'Unexpected type for value: {value} (type: {type(value)})')
#     return new_column_types

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

def is_number(value):
    """Check if a string represents a number (integer or float)."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def update_numeric_columns(row, column_names, numeric_indices, new_column_types):
    """Check numeric columns and update the column types accordingly."""
    print(f'Column names: {column_names}')
    row_list = [row.get(col) for col in column_names]
    print(f'numeric_indices: {numeric_indices}')
    for idx in numeric_indices:
        value = row_list[idx]
        print(f'Verifying column {idx}: {value} (type: {type(value)})')
        if value is None:
            continue  # Skip null values
        print(f'Checking value: {value} (type: {type(value)})')
        if isinstance(value, (int, float)):
            if float(value) % 1 != 0:
                colname = column_names[idx]
                new_column_types[colname] = Float
        elif isinstance(value, str):
            val_str = value.strip()
            print(f'{val_str} is a string.')
            if not val_str.isdigit() and is_number(val_str):
                print(f'{val_str} is a number string and not an integer.')
                if float(val_str) % 1 != 0:
                    colname = column_names[idx]
                    print(f'Not divisible by 1, setting {colname} to Float.')
                    new_column_types[colname] = Float
            else:
                print(f'{val_str} was not converted to a float.')
    return new_column_types

dataset_info = dataset_info_dict['census_blocks2020']
filename=dataset_info.dataset_path
column_types = dataset_info.col_types

# def set_dtypes(filename, column_types):
"""Main function to process the file and set data types."""
print(f'Starting {filename}')
if not filename.endswith('.json'):
    print(f'{filename} is not a json file, skipping.')
    # return column_types

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
import jsonlines
with jsonlines.open(filename, mode='r') as reader:
    for row in reader:
        # If each row comes as a list, convert it into a dict using the ordering in column_names.
        if isinstance(row, list):
            row = {col_name: (row[i] if i < len(row) else None)
                    for i, col_name in enumerate(column_names)}

        # Convert the values for known types.
        row = convert_row_values(row, column_types, column_names)

        # Update numeric column types.
    print(row)
    new_column_types = update_numeric_columns(row, column_names, numeric_indices, new_column_types)
    print(new_column_types)

    # return new_column_types













#################
for dataset_info in dataset_info_dict.values():
    # column_types = dataset_info.col_types
    dataset_info.col_types = set_dtypes(filename=dataset_info.dataset_path, column_types = dataset_info.col_types)


import pprint
pprint.pprint(dataset_info_dict)


# * #### The MapPLUTO dictionary contains most of the information we need to interpret various codes and categories meaningfully.
# * #### Unfortunately, it is in PDF format (as are many of the data dictionaries on NYCOpenData), which made extracting all the relevant data a real pain, and I don't expect most of these functions will be fully reusable for other PDFs I may encounter in the future. My hope is that it will still give me a head start when I need to make custom functions for future PDFs


# filename = '/home/james/Massive/PROJECTDATA/nyc_real_estate_data/dictionaries/mapPLUTO_data_dictionary.pdf'
filename = f"{PROJECT_DATA}/dictionaries/mapPLUTO_data_dictionary.pdf"


# * Looking at the PLUTO data dictionary, it seems that most category variables are labeled as "alpahnumeric" even if they only contain numbers, such as zip codes.
# * There are some exceptions, police precincts and districts are numeric and listed as such. However as there a limited number of repeating variables, I wil treat them as categorical as well.


pdf_by_section = map_pdf(filename, same_line_tolerance=0.3, start_page=3) 


category_markers = ['code', 'category', 'class', 'district', 'precinct', 'company', 'name', 'health_area', 'type', 'borough', 'name', 'health_area', 'health_center_district', 'overlay']

column_customizations=[]


for section in pdf_by_section:
    column_customizations += parse_pluto_dict_sections(section, category_markers)



# * Add tables from appendixes


table_dicts = parse_zoning(filename)



# Preprocess dictionary keys by truncating last letter (for singular/plural matching)
truncated_keys = {key[:-1]: value for key, value in table_dicts.items()}

# Create a sorted list of `new_name` for efficient prefix search
sorted_new_names = sorted(item.new_name for item in column_customizations)
col_customization_dict = {item.new_name: item for item in column_customizations}

# Apply updates
for key, value in truncated_keys.items():
    print(key)
    matches = find_matching_keys(key, sorted_new_names)
    print(matches)
    for match in matches:
        col_customization_dict[match].definitions = value  # Update definitions
        col_customization_dict[match].is_category = True



# Manually set BBL to not be a category
col_customization_dict['borough_tax_block_and_lot'].is_category = False


# # Parse Appendix D:
# ### Extract the last table, which isn't actually a table, just text arranged in a table-like way.


# Example usage
header_x_thresh = 10
header_y_thresh = 20
body_x_thresh = 10
body_y_thresh = 10
column_gap_thresh = 20  # Adjust based on observed spacing
ncol = 3

with pdfplumber.open(filename) as pdf:
    words = pdf.pages[-1].extract_words()  # Extract words from page 0
    merged_rows = merge_words_into_rows(words, header_x_thresh, header_y_thresh, body_x_thresh, body_y_thresh)


last_table = []
for idx,row in enumerate(merged_rows[1:]):
    new_row = []
    for idx2,cell in enumerate(row[0]):
        new_row.append(merge_text_in_cell(cell))
    last_table.append(new_row)


col_customization_dict['land_use_category'].definitions = last_table


# ### Get explanations of zoning codes.
# * I could only find this information in pdf form.
# * I discovered how hard PDFs can be to parse.
# * I had to do a lot of customization for just this specific pdf. I could have just manually cut and pasted the data from the pdf in the amount of time it took me to do that.
# * I still think it was good to do for reproducibility reasons, but in the future I will try to avoid working with datasets that have important information only in PDF format.
# * The following functions extract the tables from the pdf, detecting footnotes, and then subsitute the foonote number for the footnote text within the dataframe (so that it will end up as part of the relevant record in the databasee).


url = "https://www.nyc.gov/assets/bronxcb8/pdf/zoning_table_all.pdf"
filename = "zoning_table_all.pdf"  # Path to save the pdf containing the info we need

downloader(
            url=url,
            download_path=f"{PROJECT_DATA}/dictionaries/",
            outfile_name=filename,
            bigfile=False,
        )


# * Run the above functions to extract the data from the pdf.


tables_and_footnotes = parse_zoning_details(f"{PROJECT_DATA}/dictionaries/{filename}")


tables_and_footnotes


for tablename in tables_and_footnotes.keys():
    print(tablename)
    df = tables_and_footnotes[tablename]['df']
    df.name = df.index.name
    # with engine.connect() as conn:
    for series_name, series in df.items():
        tdf = pd.DataFrame(series)
        tdf.reset_index(inplace=True)
        jstring = pd.DataFrame(tdf).to_json()
        col_customization_dict['zoning_district_1'].definitions.append([series_name, jstring])



# ### The PDF parsed above still has some definitions that are in text outside the tables. From `zoning_table_all.pdf`:
# 
# >C1-1 through C1-5 and C2-1 through C2-5 are commercial districts which are mapped as overlays within residential districts. When a commercial overlay is mapped within an R1 through R5 district, except an R5D district, the commercial FAR is 1.0; within an R5D district or an R6 through R10 district, the commercial FAR is 2.0. The residential FAR for a commercial overlay district is determined by the residential district regulations.
# 
# * I need to manually create the object to hold this information and put it in the database


more_zones = {}
info = "Commercial districts which are mapped as overlays within residential districts. When a commercial overlay is mapped within an R1 through R5 district, except an R5D district, the commercial FAR is 1.0; within an R5D district or an R6 through R10 district, the commercial FAR is 2.0. The residential FAR for a commercial overlay district is determined by the residential district regulations."
for i in range(1,6):
    more_zones[f'C1-{i}'] = info
    more_zones[f'C2-{i}'] = info


for key in more_zones.keys():
    col_customization_dict['commercial_overlay_1'].definitions.append([key, more_zones[key]])


# ### Get a few more code meanings 
# * From [NYC Department of Tax and Finance Data Dictionary](https://www.nyc.gov/assets/finance/downloads/tar/tarfieldcodes.pdf):
#     * LandUse
#     * OwnerType
#     * Easment code
# * Additional information about commercial zoning that I have not included can be [found here](https://www.nyc.gov/assets/planning/download/pdf/zoning/districts-tools/commercial_zoning_data_tables.pdf).
# * Additional information about residential zoning that I have not included can be [found here](https://www.nyc.gov/assets/planning/download/pdf/zoning/districts-tools/residence_zoning_data_tables.pdf)


# ## Get the meanings of the building classification codes from the City of New York website.


# import urllib.request #, urllib.parse, urllib.error
# from bs4 import BeautifulSoup

webpage = "https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html"

trs = get_table_rows(webpage)

class_codes = []
d = None
for tr in trs:    
    # Check if 'a' with 'name' exists
    a = tr.find('a', attrs={'name': True})
    if a:
        if d:
            class_codes.append(d)
        supercategory = tr.find_all('th')[1].text.capitalize()
        d = {"supercategory": supercategory}
    
    # Check if 'td' exists and update 'd'
    cells = tr.find_all('td')
    if cells:
        d = {}
        code, name = cells[:2]
        d['code'] = code.text.strip()
        d['name'] = name.text.capitalize().strip()
        class_codes.append(d)



for row in class_codes:
    col_customization_dict['building_class'].definitions.append([row['code'], row['name']])


dataset_info_dict['mapPLUTO'].col_customizations = col_customization_dict


for name,info in dataset_info_dict.items():
    # print(info.col_customizations)
    if info.col_types.items():
        if not info.col_customizations:
            info.col_customizations = {short_name : ColCustomization(short_name=short_name, dtype=dtype) for short_name,dtype in info.col_types.items()}
        for key,val in info.cardinality_ratios.items():
            if val > 20 and info.col_customizations is not None and info.col_types[key] == String:
                info.col_customizations[key].is_category = True



with open("environment_data/table_dicts.pkl", "wb") as f:
    dill.dump(
        {
            "dataset_info_dict": dataset_info_dict,
            "PROJECT_PATH": PROJECT_PATH,
            "PROJECT_DATA": PROJECT_DATA,
            "SQLITE_PATH": SQLITE_PATH,
            "DATADIR": DATADIR,
            "PROJECT_NAME": PROJECT_NAME,
            "PROJECT_DATA": PROJECT_DATA,
        },
        f,
    )


