#!/usr/bin/env python
# coding: utf-8

# In[1]:


datasets = {'primary-land-use-tax-lot-output-pluto': '64uk-42ks',
            'city-government/property-valuation-and-assessment-data': 'yjxr-fw8i',
            'tax-lien-sale-lists': '9rz4-mjek',
            'housing-maintenance-code-violations': 'wvxf-dwi5',
            'assessment-actions': '4nft-bihw',
            'housing-database': '6umk-irkx'}


# In[2]:


# Directory where the data folder for this analysis is to be created
datadir = "/home/james/Massive/PROJECTDATA"

# Name of the folder in which the project data is stored
project_name = "nyc_real_estate"


# In[3]:


# Import standard libraries
import os
import re
import json
import dataclasses
import codecs
import requests
from urllib.request import urlopen
import datetime


# # Import third-party libraries
import geopandas as gpd
from geoalchemy2 import Geometry
import pandas as pd
import numpy as np
import pyogrio
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Float, String, Date, MetaData, event, Table, text, LargeBinary, ForeignKey
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql.sqltypes import Boolean
from sqlalchemy.event import listen
from sqlalchemy.engine import Engine
# from sqlalchemy.ext.declarative import declarative_base

import sqlite3
import fiona
from fiona.crs import from_epsg



# In[4]:


import dataclasses

@dataclasses.dataclass
class Dataset:
    """Class to hold dataset metadata"""
    id: str
    main_url: str
    data_url: str
    data_dict_url: str
    other_files: list
    name: str
    attribution: str
    createdAt: str
    description: str
    provenance: str
    publicationDate: str
    rowsUpdatedAt: str
    columns: dict


# In[6]:


project_path = os.getcwd()

PROJECTDATA = f"{datadir}/{project_name}_data"
sqlite_path = f'sqlite:///{PROJECTDATA}/{project_name}_db.sqlite'


# Create necessary directories
os.makedirs(PROJECTDATA, exist_ok=True)
os.makedirs(f"{PROJECTDATA}/Downloads", exist_ok=True)
os.makedirs(f"{PROJECTDATA}/intermediate_files", exist_ok=True)

# Set environment variables
os.environ["PROJECTDATA"] = PROJECTDATA
os.environ["project_name"] = project_name
os.chdir(project_path)


# In[7]:


# Define shared dataset configurations
shared_dataset_configs = {
    "prefix": PROJECTDATA,
    "cols_to_drop": ["id", "sid", "position", "created_at", "created_meta", "updated_at", "updated_meta", "borough", "meta"],
    "cols_to_rename": {},
    "lookup_columns": [],
    "dtype_exceptions": {'zip_code': String, "meta_data": String, "postcode": String, "calendar_date": Date, "number": Integer, "text": String, "point": String}
}

# Define specific dataset configurations
specific_dataset_configs = {
    "lien_data": {
        "prefix": f'{PROJECTDATA}/intermediate_files',
        "cols_to_drop": [],
        "cols_to_rename": {'BORO': 'borough'},
        "lookup_columns": [],
        "dtype_exceptions": {}
    },
    "assessment_data": {
        "prefix": f'{PROJECTDATA}/intermediate_files',
        "cols_to_drop": [],
        "cols_to_rename": {"BLDGCL": "building_class", "TAXCLASS": "tax_class_code", "Zip Codes": "zip_code"},
        "lookup_columns": ["building_class", "street_name", "owner", "zip_code"],
        "dtype_exceptions": {}
    }
}


# In[ ]:


# engine = create_engine(f'{sqlite_path}?check_same_thread=False', echo=False)

# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# In[9]:


import nbformat
from IPython.display import display, Javascript
import ipynbname

# Auto-save the notebook before extracting cells
display(Javascript("IPython.notebook.save_checkpoint()"))

# Get current notebook name dynamically
notebook_name = ipynbname.name() + ".ipynb"

# Read the notebook file
with open(notebook_name, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Extract all cells except the last one
code_cells = [cell["source"] for cell in nb.cells[:-1] if cell["cell_type"] == "code"]

# Write extracted code to a .py file
with open("shared_code.py", "w") as f:
    f.write("\n\n".join(code_cells))

print("✅ Saved all but the last cell to shared_code.py")


# In[11]:


ipynbname.name()


# In[12]:


import os
import json

def get_notebook_name():
    try:
        # Get the connection file path
        connection_file = os.path.basename(get_ipython().config['IPKernelApp']['connection_file'])
        # Get the kernel ID
        kernel_id = connection_file.split('-', 1)[1].split('.')[0]
        # Get the notebook server sessions
        sessions = json.load(open(os.path.join(os.path.dirname(connection_file), 'nbserver-{}.json'.format(kernel_id))))
        # Find the notebook name
        for session in sessions['sessions']:
            if session['kernel']['id'] == kernel_id:
                return os.path.basename(session['notebook']['path'])
    except Exception as e:
        print(f"Error: {e}")
        return None

notebook_name = get_notebook_name()
print(notebook_name)


# In[ ]:




