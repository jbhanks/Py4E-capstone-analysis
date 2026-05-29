
# %%
import os
import time
from dataclasses import dataclass, field
from typing import List
import re
import requests
from sqlalchemy import Integer, Float, String, Date, LargeBinary, Boolean
import dill
from src.helpers import *
from src.models import *

######
import faulthandler
import os
import signal
import sys
import traceback
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlencode

faulthandler.enable()

def dump_and_exit(signum, frame):
    signame = signal.Signals(signum).name
    print(f"\nRECEIVED {signame} in pid={os.getpid()}", file=sys.stderr, flush=True)
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(128 + signum)

for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(sig, dump_and_exit)

print(f"step1.py pid={os.getpid()}", flush=True)

################

# %% [markdown]
# - ### Customize the directories to be used to where on your machine you want them.
#   - #### Set paths for where to create the data folder where datasets will be downloaded and the sqlite database created. It will be at least several gigabytes if not more, so make sure you have space.
# 

# %%
# Directory where the data folder for this analysis is to be created
DATADIR = "/home/james/Massive/PROJECTDATA/"

# Name of the folder in which the project data is stored
PROJECT_NAME = "nyc_real_estate"

# MapPluto version, check website for latest
pluto_version = "25v2_1"

# %% [markdown]
# - ### Set the names and paths of the datasets to be used. Here they are already set for the analysis I am doing, but the hope is to make it as adaptable to other datasets as possible.
# 

# %%
# Create paths that will be frequently used throughout the project
PROJECT_PATH = os.getcwd()
PROJECT_DATA = f"{DATADIR}/{PROJECT_NAME}_data"
SQLITE_PATH = f"sqlite:///{PROJECT_DATA}/{PROJECT_NAME}_db.sqlite"

# %% [markdown]
# * Create the directories

# %%

# Create necessary directories from the paths
os.makedirs(
    PROJECT_DATA, exist_ok=True
)  # Make the main directory for the project downloads and data
os.makedirs(f"{PROJECT_DATA}/downloads", exist_ok=True)  # Make the download directory
# os.makedirs(
#     f"{PROJECT_DATA}/intermediate_files", exist_ok=True
# )  # Location for intermediate files
os.makedirs(
    f"{PROJECT_DATA}/files_to_use", exist_ok=True
)  # Location where cleaned data to be directly used in analysis is stored
os.makedirs(
    f"{PROJECT_DATA}/dictionaries", exist_ok=True
)  # Location where cleaned data to be directly used in analysis is stored
os.makedirs(
    f"{PROJECT_DATA}/figures", exist_ok=True
)  # Location where plots will go later on

# %% [markdown]
# - ### Create a dataclass to store metadata about the datasets.
# 

# %% [markdown]
# - ### Create instances of the new dataclass for each dataset.
# 
#   - #### Create a dictionary of instances of the `Dataset` dataclass, one for each dataset I have chosen The keys are a shortened version of the dataset name that I made up. Values are the urls for the information pages that one finds when [searching the OpenData website](https://data.cityofnewyork.us/browse?q=&sortBy=relevance). Although it would take a lot of testing to verify, the intention is to be able to substitute a different set of urls and proceed through the notebook set with minimal modification. I opted not to use it this time, but if the code proves to be reusable, I could use a tool like [Papermill](https://papermill.readthedocs.io/en/latest/)
# 

# %%
dataset_info_dict = {
    "mapPLUTO": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-Map-MapPLUTO-/f888-ni5f/about_data",
        standard=False,
        data_url=f"https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/mappluto/nyc_mappluto_{pluto_version}_fgdb.zip",
        format="zip",
        geodata=True,
    ),
    "property_address_directory": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Property-Address-Directory/bc8t-ecyu/about_data",
        standard=False,
        data_url=f"https://data.cityofnewyork.us/download/bc8t-ecyu/application%2Fzip",
        format="zip",
        geodata=False,
    ),
    "pseudo_lots": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Pseudo-Lots/dx24-9ef7/about_data",
        standard=False,
        data_url=f"https://data.cityofnewyork.us/resource/dx24-9ef7.geojson",
        format="geojson",
        geodata=True,
    ),
    "air_quality": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r/about_data"
    ),
    "assessments": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Property-Valuation-and-Assessment-Data/yjxr-fw8i/about_data"
    ),
    "tax_liens": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Tax-Lien-Sale-Lists/9rz4-mjek/about_data"
    ),
    "housing_violations": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5/about_data"
    ),
    "assessment_actions": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Assessment-Actions/4nft-bihw/about_data"
    ),
    "DOB_violations": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Housing-Development/DOB-Violations/3h2n-5cm9/about_data"
    ),
    "DOB_stalled_construction_sites": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Housing-Development/DOB-Stalled-Construction-Sites/i296-73x5/about_data"
    ),
    "NYPD_crime_complaints": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data"
    ),
    "NYPD_summons": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Historic-/sv2w-rv3k/about_data"
    ),
    "NYPD_arrests": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data"
    ),
    "rodent_inspection": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj/about_data"
    ),
    "fire_incident_dispatch": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Public-Safety/Fire-Incident-Dispatch-Data/8m42-w767/about_data"
    ),
    "housing_database": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Housing-Development/Housing-Database/6umk-irkx/about_data",
        standard=False,
        data_url="https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/housing-project-level/nychousingdb_24q4_gdb.zip",
        format="zip",
        geodata=True,
    ),
    "NTAs2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd/about_data"
    ),
    "NTA_population_2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/New-York-City-Population-By-Neighborhood-Tabulatio/swpk-hqdp/about_data"
    ),
    "NTA_demographics_2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/Census-Demographics-at-the-Neighborhood-Tabulation/rnsn-acs2/about_data"
    ),
    "census_blocks2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/2020-Census-Blocks/wmsu-5muw/about_data"
    ),
    "CDTAs2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/2020-Community-District-Tabulation-Areas-CDTAs-/xn3r-zk6y/about_data"
    ),
    "puma2020": DatasetInfo(
        main_url="https://data.cityofnewyork.us/City-Government/2020-Public-Use-Microdata-Areas-PUMAs-/pikk-p9nv/about_data"
    ),
    "cert_of_occupancy": DatasetInfo(
        main_url="https://data.cityofnewyork.us/Housing-Development/DOB-Certificate-Of-Occupancy/bs8b-p36w/about_data"
    ),
}


# %%
def clean_name(full_name: str):
    patterns = [
        (re.compile(r"[ ,–]+", flags=re.IGNORECASE), "_"),
        (re.compile(r"#", flags=re.IGNORECASE), "num"),
        (re.compile(r"/", flags=re.IGNORECASE), "_or_"),
        (re.compile(r"&", flags=re.IGNORECASE), "and"),
        (re.compile(r"!(altered)_[0-9]$", flags=re.IGNORECASE), ""),
    ]
    new_name = full_name.lower()
    for pattern, replacement in patterns:
        new_name = pattern.sub(replacement, new_name)
    return new_name


# %%
def parse_metadata(metadata):
    try:
        column_metadata = metadata["columns"]
        metadata = {k: v for k, v in metadata.items() if k != "columns"}
        for column in column_metadata:
            if "cachedContents" in column:
                column["cachedContents"].pop("top", None)
        column_metadata = column_metadata
        cardinality_ratios = {
            clean_name(column["name"]): int(column["cachedContents"]["non_null"])
            / int(column["cachedContents"]["cardinality"])
            for column in column_metadata
            if "cachedContents" in column.keys()
            and int(column["cachedContents"].get("cardinality", 0)) != 0
        }
        for col in column_metadata:
            print(f"Column: {col}")
        col_types = {
            clean_name(col["name"]): col["dataTypeName"]
            for col in column_metadata
            if column_filter(col)
        }
        print(f"col_types: {col_types}")
        return {
            "metadata": metadata,
            "column_metadata": column_metadata,
            "cardinality_ratios": cardinality_ratios,
            "col_types": col_types,
        }
    except Exception as e:
        print(f"No column metadata in {metadata}")
        raise e


# %%
def is_socrata_rows_json_url(url):
    return (
        isinstance(url, str)
        and "data.cityofnewyork.us/api/views/" in url
        and "/rows.json" in url
        and "accessType=DOWNLOAD" in url
    )


def validate_json_with_jq(path):
    path = Path(path)

    if not path.exists():
        return False

    if shutil.which("jq") is None:
        # Fallback: avoid json.load() on multi-GB files. This is only a weak
        # completion check, but the paged writer uses .tmp + atomic replace.
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 4096))
            tail = f.read().rstrip()
        return tail.endswith(b"]}") or tail.endswith(b"]}\n")

    result = subprocess.run(
        ["jq", "empty", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"Invalid JSON file: {path}", flush=True)
        print(result.stderr[-2000:], flush=True)
        return False

    return True


def page_cache_is_valid(path):
    path = Path(path)

    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list)
    except Exception:
        return False


def request_json_with_retries(session, url, headers=None, timeout=(30, 300), attempts=5):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            print(
                f"Request failed on attempt {attempt}/{attempts}: {url}\n{type(e).__name__}: {e}",
                flush=True,
            )
            time.sleep(min(60, 2 ** attempt))

    raise last_error


def download_socrata_json_paged(
    dataset_info,
    download_path,
    outfile_name,
    page_size=25_000,
    sleep_seconds=0.5,
):
    download_dir = Path(download_path)
    download_dir.mkdir(parents=True, exist_ok=True)

    out = download_dir / outfile_name
    tmp_out = out.with_suffix(out.suffix + ".tmp")
    page_dir = download_dir / "_pages" / out.stem
    page_dir.mkdir(parents=True, exist_ok=True)

    if out.exists():
        if validate_json_with_jq(out):
            print(f"File already exists and is valid JSON: {out}", flush=True)
            return str(out)

        bad_path = out.with_suffix(out.suffix + f".bad-{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"Existing JSON is invalid; moving it to {bad_path}", flush=True)
        out.rename(bad_path)

    if tmp_out.exists():
        print(f"Removing stale temp file: {tmp_out}", flush=True)
        tmp_out.unlink()

    headers = {
        "User-Agent": "nyc-real-estate-capstone/1.0",
        "Accept": "application/json",
    }

    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token

    session = requests.Session()

    print(f"Starting paged Socrata download: {dataset_info.short_name}", flush=True)

    metadata = request_json_with_retries(
        session,
        dataset_info.metadata_url,
        headers=headers,
        timeout=(30, 120),
    )

    columns = [
        col
        for col in metadata["columns"]
        if col.get("fieldName") and not col["fieldName"].startswith(":")
    ]

    metadata = dict(metadata)
    metadata["columns"] = columns
    field_names = [col["fieldName"] for col in columns]

    offset = 0
    total_rows = 0

    while True:
        page_path = page_dir / f"offset_{offset:012d}.json"
        tmp_page_path = page_path.with_suffix(".json.tmp")

        if page_cache_is_valid(page_path):
            with page_path.open("r", encoding="utf-8") as f:
                rows = json.load(f)

            print(
                f"Using cached page offset={offset:,} rows={len(rows):,}",
                flush=True,
            )
        else:
            params = urlencode(
                {
                    "$limit": page_size,
                    "$offset": offset,
                    "$order": ":id",
                }
            )
            url = f"https://data.cityofnewyork.us/resource/{dataset_info.id}.json?{params}"

            print(f"Downloading offset={offset:,}", flush=True)

            rows = request_json_with_retries(
                session,
                url,
                headers=headers,
                timeout=(30, 300),
            )

            if not isinstance(rows, list):
                raise RuntimeError(f"Expected a list of rows from {url}, got {type(rows).__name__}: {rows}")

            if rows:
                with tmp_page_path.open("w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False)

                tmp_page_path.replace(page_path)

        if not rows:
            print("No more rows.", flush=True)
            break

        total_rows += len(rows)
        print(f"Total rows cached: {total_rows:,}", flush=True)

        if len(rows) < page_size:
            break

        offset += page_size
        time.sleep(sleep_seconds)

    print(f"Assembling final file: {tmp_out}", flush=True)

    assembled_rows = 0

    with tmp_out.open("w", encoding="utf-8") as f:
        f.write('{"meta":{"view":')
        json.dump(metadata, f, ensure_ascii=False)
        f.write('},"data":[\n')

        first = True

        for page_path in sorted(page_dir.glob("offset_*.json")):
            with page_path.open("r", encoding="utf-8") as page_file:
                rows = json.load(page_file)

            for row in rows:
                row_values = [row.get(field) for field in field_names]

                if not first:
                    f.write(",\n")

                json.dump(row_values, f, ensure_ascii=False)
                first = False
                assembled_rows += 1

        f.write("\n]}\n")

    tmp_out.replace(out)

    print(f"Finished paged Socrata download: {out}", flush=True)
    print(f"Rows assembled: {assembled_rows:,}", flush=True)

    return str(out)

# %% [markdown]
# - ### Inspect the metadata for the chosen datasets to get column names, data types, and cardinality for each column.
# 

# %%
for short_name, dataset_info in dataset_info_dict.items():
    print(f"Getting metadata for {short_name}... {dataset_info.metadata_url}")
    response = requests.get(dataset_info.metadata_url, timeout=10)
    response.raise_for_status()
    metadata = response.json()
    data_dict = [
        (attachment["assetId"], attachment["filename"])
        for attachment in metadata["metadata"]["attachments"]
        if any(s in attachment["filename"].casefold() for s in ["dict", "dd"])
    ]
    if data_dict:
        dataset_info.data_dict_url = f"https://data.cityofnewyork.us/api/views/{dataset_info.id}/files/{data_dict[0][0]}?download=true&filename={data_dict[0][1]}"
    dataset_info.name = metadata["name"]
    dataset_info.short_name = short_name
    try:
        dataset_info.other_files = [
            (attachment["assetId"], attachment["filename"])
            for attachment in metadata["metadata"]["attachments"]
            if not any(s in attachment["filename"].casefold() for s in ["dict", "dd"])
        ]
    except KeyError:
        dataset_info.other_files = None
    try:
        dataset_info.attribution = metadata["attribution"]
    except KeyError:
        dataset_info.attribution = None
    try:
        dataset_info.createdAt = metadata["createdAt"]
    except KeyError:
        dataset_info.attribution = None
    try:
        dataset_info.description = metadata["description"]
    except KeyError:
        dataset_info.attribution = None
    try:
        dataset_info.provenance = metadata["provenance"]
    except KeyError:
        dataset_info.attribution = None
    try:
        dataset_info.publicationDate = metadata["publicationDate"]
    except KeyError:
        dataset_info.attribution = None
    try:
        dataset_info.rowsUpdatedAt = (
            (  # The column for when the dataset was last updated is usually called `rowsUpdatedAt`, but sometimes it's called `viewLastModified`
                metadata["rowsUpdatedAt"]
                if "rowsUpdatedAt" in metadata.keys()
                else (
                    metadata["viewLastModified"]
                    if "viewLastModified" in metadata.keys()
                    else None
                )
            ),
        )
    except KeyError:
        dataset_info.attribution = None
    time.sleep(5)

# %%
dataset_info_dict.keys()
dataset_info_dict['pseudo_lots']

# %% [markdown]
# * ### Download the datasets

# %%
for name, dataset_info in dataset_info_dict.items():
    print(name, flush=True)
    print(dataset_info.data_url, flush=True)

    outfile_name = f"{dataset_info.short_name}.{dataset_info.format}"

    if dataset_info.format == "json" and is_socrata_rows_json_url(dataset_info.data_url):
        download_socrata_json_paged(
            dataset_info=dataset_info,
            download_path=f"{PROJECT_DATA}/downloads/",
            outfile_name=outfile_name,
            page_size=25_000,
            sleep_seconds=0.5,
        )
    else:
        downloader(
            url=dataset_info.data_url,
            download_path=f"{PROJECT_DATA}/downloads/",
            outfile_name=outfile_name,
            bigfile=True,
            chunk_size=1024 * 1024,  # 1 MB
        )

    if dataset_info.data_dict_url is not None:
        try:
            file_type = dataset_info.data_dict_url.split(".")[-1]
        except AttributeError:
            print(
                f"{RED}Could not parse filetype from {CYAN}{dataset_info.data_dict_url}{RESET}!"
            )
            continue
        print(dataset_info.data_dict_url)
        dataset_info.data_dict_path = downloader(
            url=dataset_info.data_dict_url,
            download_path=f"{PROJECT_DATA}/dictionaries/",
            outfile_name=f"{dataset_info.short_name}_data_dictionary.{file_type}",
            bigfile=False,
        )

# %%
dataset_info_dict.items()

# %% [markdown]
#   * Filter the json datasets to remove unneeded data and unzip any zipped files.

# %%
for name, dataset_info in dataset_info_dict.items():
    print(name)
    if dataset_info.format == "json":
        infile = f"{PROJECT_DATA}/downloads/{name}.json"
        print("The infile is:", infile)
        metadata = jq_metadata(infile, metadata_filter=".meta.view")
        vars(dataset_info).update(parse_metadata(metadata))
        data_filter = make_jq_filter(metadata['columns'])
        dataset_info.dataset_path = jqfilter(
            infile=infile,
            outfile=f"{PROJECT_DATA}/files_to_use/{name}_rows.json",
            data_filter=data_filter
        )
    elif dataset_info.format == "zip":
        print(dataset_info)
        dataset_info.dataset_path = unzipper(
            zip_path=f"{PROJECT_DATA}/downloads/{name}.zip",
            outdir=f"{PROJECT_DATA}/files_to_use/",
            dict_dir=f"{PROJECT_DATA}/dictionaries/",
            extension=".gdb",
        )
    elif dataset_info.format == "geojson":
        infile = f"{PROJECT_DATA}/downloads/{name}.geojson"
        print("The infile is:", infile)
        dataset_info.dataset_path = infile

# %%
import fiona

for name, dataset_info in dataset_info_dict.items():
    print(f"checking {name}")
    dataset_path = getattr(dataset_info, "dataset_path", None)

    if dataset_path is None:
        print(f"No dataset_path for {name}; skipping geodatabase inspection")
        continue

    if str(dataset_path).endswith('.gdb'):
        print(f'{name} is a gdb file')
        gdb_path = dataset_path
        layers = fiona.listlayers(gdb_path)
        d = {}
        for layer in layers:
            with fiona.open(gdb_path, layer=layer) as source:
                d = d | {key.lower(): val for key, val in source.schema['properties'].items()}
        dataset_info.col_types = d


# %%
print(dataset_info_dict['mapPLUTO'].col_types)

# %%
print(dataset_info_dict)

# %% [markdown]
# * Identify potential synonyms

# %%
from collections import defaultdict


data = {name : list(dataset.col_types.keys()) for name,dataset in dataset_info_dict.items()}

results = defaultdict(list)

for category, items in data.items():
    for item in items:
        results[item].append(category)

# Convert defaultdict to a regular dict
results = dict(results)

results = dict(sorted(results.items()))

for result in results.items():
    print(result)


# %% [markdown]
# * Make a list of lists of synonyms used by the column names of different datasets. The first name in the list is what all matching column names will be standardized to.

# %%
synonyms_list = [
    ["address", "property_address", "staddr"],
    ['house_number', 'addressnum', 'housenumber'],
    ['community_board', 'cd', 'commntydst', 'communityboard'],
    ["building_class", "bldgcl", "bldgclass", "bldg_class"],
    ["block_number", "block", "block_"],
    ['census_tract', 'censustract', 'centract20', 'ct2020'],
    ["borough_code", "borocode", "boroughcode", "boro", "boroid"], # Hopefully I won't come across a dateset where the borough name is called "BORO"
    ["borough", "boroughname", "boroname"],
    ["council_district", "council", "councildst", "councildistrict"],
    ['cdta_name', 'cdtaname', 'cdta2020', 'cdtaname20'],
    ['police_precinct', 'policeprct', 'policepcnt'],
    ["exempttot", "extot"],
    ["lot_number", "lot"],
    ["number_of_floors", "numfloors", "stories"],
    ["owner_name", "ownername", 'owner'],
    ['owner_type', 'ownership', 'ownertype'],
    ["zip_code", "postcode", "zipcode", "zip"],
    ["street_name", "street", "streetname"],
    ["tax_class_code", "taxclass"],
    ['special_purpose_district_1', 'spdist1', 'specldst1'],
    ['special_purpose_district_2', 'spdist2', 'specldst2'],
    ['special_purpose_district_3', 'spdist3', 'specldst3'],
    ['zoning_district_1', 'zonedist1', 'zoningdst1'],
    ['zoning_district_2', 'zonedist2', 'zoningdst2'],
    ['zoning_district_3','zonedist3', 'zoningdst3'],
    ['lot_depth', 'lotdepth', 'ltdepth'],
    ['lot_front', 'lotfront', 'ltfront'],
    ['nta', 'nta_name', 'ntaname', 'ntaname20']
]

# %%
def rename_keys(d, synonyms_list):
    """
    Given a dictionary d, returns a new dictionary where each key that appears
    in any group of synonyms in synonyms_list is replaced by the first synonym.
    """
    new_d = {}
    for key, value in d.items():
        new_key = key
        for synonyms in synonyms_list:
            if key in synonyms:
                new_key = synonyms[0]
                break
        new_d[new_key] = value
    return new_d




for name, dataset_info in dataset_info_dict.items():
    dataset_info.col_types = rename_keys(dataset_info.col_types, synonyms_list)
    for colname in dataset_info.col_types:
        if colname not in dataset_info.col_customizations.keys():
            dataset_info.col_customizations[colname] = ColCustomization(short_name=colname)  # or a default dictionary/object
    dataset_info.cardinality_ratios = rename_keys(dataset_info.cardinality_ratios, synonyms_list)
    catcols = [colname for colname,ratio in dataset_info.cardinality_ratios.items() if ratio > 25 and dataset_info.col_types[colname] == 'text']
    for colname in catcols:
        dataset_info.col_customizations[colname].is_category = True

# %% [markdown]
# * Inspect the data that has been compiled about each dataset.

# %%
dataset_info_dict

# %%
# objects_to_save = [datasets, PROJECT_PATH, PROJECT_DATA, SQLITE_PATH, DATADIR, PROJECT_NAME]
os.makedirs(
    f"environment_data", exist_ok=True
)

objects_to_save = {
    "dataset_info_dict": dataset_info_dict,
    "PROJECT_PATH": PROJECT_PATH,
    "PROJECT_DATA": PROJECT_DATA,
    "SQLITE_PATH": SQLITE_PATH,
    "DATADIR": DATADIR,
    "PROJECT_NAME": PROJECT_NAME,
    "PROJECT_DATA": PROJECT_DATA,
    "PLUTO_VERSION":  pluto_version,
}

with open("environment_data/select.pkl", "wb") as f:
    dill.dump(objects_to_save, f)

# %%



