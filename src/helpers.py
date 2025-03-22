import re
import json
import requests
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import logging
import subprocess
from collections import defaultdict
import glob
import os
from dateutil.parser import parse


# ANSI color codes
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
WHITE = "\x1b[37m"
RESET = "\x1b[0m"


def get_file(url, session, headers, retryParams, stream=False):
    session.mount(url, HTTPAdapter(max_retries=retryParams))
    with session.get(
        url, stream=stream, timeout=random.uniform(10, 12), headers=headers
    ) as response:
        if response.status_code == 200:
            if "<title>Access Denied</title>" in response.text:
                raise Exception(f"{RED}Access was denied for {CYAN}{url}{RESET}")
            else:
                return response
        else:
            raise Exception(
                f"Failed to download file {url} with status code {RED}{response.status_code}{RESET}"
            )


def downloader(url, outfile_name, download_path, bigfile=False, chunk_size=8192):
    """Downloads a file from a url

    Args:
        url (str): url of the file
        outfile_name (str): filename for the downloaded file
        download_path (str): path of the save location
        bigfile (bool, optional): Download and write in chunks so as not to keep entire download in memory. Defaults to False.
        chunk_size (int, optional): Chunk size for if bigile is True. Defaults to 8192.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    # Log the downloads
    logging.basicConfig(
        filename=f"{download_path}/big_downloads.log",
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)
    retryParams = Retry(
        total=3,
        backoff_factor=random.uniform(9, 12),
        status_forcelist=[408, 500, 502, 503, 504, 111, 429],
    )
    headers = {"Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"}

    file_path = os.path.join(download_path, outfile_name)

    # Skip download if file already exists
    if os.path.exists(file_path):
        print(f"{GREEN}File already exists: {CYAN}{file_path}{RESET}")
        return file_path

    with requests.Session() as session:
        print("starting download session")
        try:
            if bigfile:
                response = get_file(url, session, headers, retryParams, stream=True)
                with open(f"{download_path}{outfile_name}", "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
                    print(f"{GREEN}Success downloading {CYAN}{url}{RESET}")
            else:
                print(f"About to try {url}, will save to {download_path}{outfile_name}")
                response = get_file(url, session, headers, retryParams, stream=False)
                with open(f"{download_path}{outfile_name}", "wb") as file:
                    file.write(response.content)
                    print(f"{GREEN}Success downloading {CYAN}{url}{RESET}")
        except requests.exceptions.ReadTimeout as e:
            logger.error(f"Request for {url} failed with read timeout: {RED}{e}{RESET}")
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Request for {url} failed with a connection error: {RED}{e}{RESET}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Request for {url} failed with a request exception: {RED}{e}{RESET}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred: {RED}{e}{RESET}")
            raise Exception(e)
        return file_path


def parse_metadata(metadata):
    try:
        columns = metadata["columns"]
        column_metadata = {column["fieldName"]: column for column in columns}
        cardinality_ratios = {
            column["fieldName"]: int(column["cachedContents"]["non_null"])
            / int(column["cachedContents"]["cardinality"])
            for column in columns
            if "cachedContents" in column.keys()
        }
        col_types = {col["fieldName"]: col["dataTypeName"] for col in columns}
        return column_metadata, cardinality_ratios, col_types
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return None, None, None


def jqfilter(infile, outfile, metadata_filter=".meta.view.columns"):
    """Filters json datasets from NYC OpenData to remove metadata columns based on the metadata included in the dataset

    Args:
        infile (str): json file to filter
        outfile (str): filtered json output path
        metadata_filter (str, optional): jq selector for column metadata. Defaults to '.meta.view.columns'.

    Returns:
        str: Path of the filtered json file
    """
    # First, extract metadata (that output is expected to be small)
    print(f'Filtering {infile}')
    with open(infile, "r") as f:
        metadata = json.loads(
            subprocess.run(
                ["jq", metadata_filter],
                stdin=f,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    print(f'Got metadata for {infile}')
    selected_indices = [
        i for i, col in enumerate(metadata) if col["renderTypeName"] != "meta_data" or not col["fieldName"].startswith(':@')
    ]  # Filter out metadata columns
    print(f'selected_indices are {selected_indices}')
    data_filter = f'[.data[] | [{", ".join(f".[{i}]" for i in selected_indices)}]]'

    with open(infile, "r") as inf, open(outfile, "w") as outf:
        subprocess.run(
            ["jq", data_filter],
            stdin=inf,
            stdout=outf,
            check=True,  # will raise an exception on error
        )
    fd = outf.fileno()
    os.fsync(fd) 
    print(f"Output written to: {outfile}")
    return outfile


def move_data(path, dict_dir):
    dict_files = glob.glob(os.path.join(path, "*.xlsx")) + glob.glob(
        os.path.join(path, "*.pdf")
    )
    if dict_files:
        print(f"dict_files: {dict_files}")
        for file in dict_files:
            subprocess.run(["mv", file, dict_dir], check=True, text=True)
        print(f"Successfully moved files to '{dict_dir}'")
    else:
        print(f"No dict files found in '{path}'")


def find_main_directory(list_result, extension=".gdb"):
    directory_sizes = defaultdict(int)

    for line in list_result.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "Length":  # Skip headers and invalid lines
            continue

        try:
            size = int(parts[0])  # First column is the file size
        except ValueError:
            continue  # Skip lines where size isn't a number

        filename = " ".join(parts[3:])  # Extract filename
        parts = filename.lstrip("/").split("/", 1)

        if len(parts) > 1:  # It's inside a directory
            top_level_dir = parts[0]

            # Check if the directory name has a desired extension
            if top_level_dir.endswith(extension):
                directory_sizes[top_level_dir] += size

    # Select the largest directory
    main_directory = max(directory_sizes, key=directory_sizes.get, default=None)
    return main_directory


def unzipper(zip_path, outdir, dict_dir, extension=".gdb"):
    """Unzips zip files containing GDB data from the NYC.gov website and places the data dictionary and gdb files in the directories expected by the project.

    Args:
        zip_path (_type_): _description_
        outdir (_type_): _description_
        dict_dir (_type_): _description_
        extension (str, optional): _description_. Defaults to ".gdb".

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    try:
        outfiles = subprocess.run(
            ["unzip", "-l", zip_path, "-d", outdir], capture_output=True, text=False
        ).stdout.decode("utf-8")
        print(f"outfile ares: {outfiles}")
        main_file = find_main_directory(outfiles, extension=extension)
        print(f"main file: {main_file}")
        subprocess.run(
            ["unzip", "-o", zip_path, "-d", outdir], capture_output=True, text=False
        )
        move_data(outdir, dict_dir)
        return f"{outdir}{main_file}"
    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception(e)


def getDatasetRowCount(url):
    """
    Gets the number of rows of a dataset under the Socrata API
    """
    import codecs
    from urllib.request import urlopen
    import json

    reader = codecs.getreader("utf-8")
    obj = reader(urlopen(url + "?$select=count(*)"))
    obj = json.load(obj)
    count = int(obj[0]["count"])
    return count


def isCategory(entry, category_markers, alphanumeric_exceptions, numeric_exceptions):
    d = {}
    for key, value in entry.items():
        if key == "Format":
            if (
                value.startswith("Alphanumeric")
                or any([marker in value for marker in numeric_exceptions])
            ) and not any([marker in value for marker in alphanumeric_exceptions]):
                d["category"] = True
            elif any([marker in entry["Field Name"] for marker in category_markers]):
                d["category"] = True
            else:
                d["category"] = False
        d[key] = value
    return d


def merge_dicts(shared: dict, specific: dict) -> dict:
    merged = {}
    for key in set(shared) | set(specific):
        shared_value, specific_value = shared.get(key), specific.get(key)

        if isinstance(shared_value, dict) and isinstance(specific_value, dict):
            merged[key] = merge_dicts(shared_value, specific_value)
        elif isinstance(shared_value, list) and isinstance(specific_value, list):
            merged[key] = shared_value + specific_value
        elif specific_value is not None:
            merged[key] = specific_value
        else:
            merged[key] = shared_value
    return merged


def parseDateString(date_string: str):
    date_object = parse(date_string, fuzzy=True)
    return date_object


# Precompiled regular expressions
_pattern_weird_start = re.compile(r"^[^a-zA-Z(#&\d:]")
_pattern_quotes = re.compile(r'"')

def textClean(text: str):
    text = text.strip()
    # Use the precompiled regex to remove unwanted starting characters
    text = _pattern_weird_start.sub("", text)
    # Remove remaining quotes
    text = _pattern_quotes.sub("", text)
    return text.strip()


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


def clean_name(full_name: str):
    patterns = [
        (re.compile(r"[ ,–]+", flags=re.IGNORECASE), "_"),
        (re.compile(r"#", flags=re.IGNORECASE), "num"),
        (re.compile(r"/", flags=re.IGNORECASE), "_or_"),
        (re.compile(r"&", flags=re.IGNORECASE), "and"),
        (re.compile(r"!(altered)_[0-9]$", flags=re.IGNORECASE), ""),
        (re.compile(r"\bboro(?!ugh)", flags=re.IGNORECASE), "borough"),
    ]
    new_name = full_name.lower()
    for pattern, replacement in patterns:
        new_name = pattern.sub(replacement, new_name)
    return new_name


# Explicitly define what gets imported when using `from models import *`
__all__ = [
    "get_file",
    "downloader",
    "jqfilter",
    "unzipper",
    "getDatasetRowCount",
    "isCategory",
    "parseDateString",
    "split_address",
    "merge_dicts",
    "textClean",
    "clean_name",
]
