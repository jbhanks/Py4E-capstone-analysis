import requests
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import logging
import subprocess


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


def downloader(url, download_path, outfile_name, bigfile=False, chunk_size=8192):
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
    with requests.Session() as session:
        try:
            if bigfile:
                response = get_file(url, session, headers, retryParams, stream=True)
                with open(f"{download_path}/{outfile_name}", "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
                    print(f"{GREEN}Success downloading {CYAN}{url}{RESET}")
            else:
                response = get_file(url, session, headers, retryParams, stream=False)
                with open(f"{download_path}/{outfile_name}", "wb") as file:
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


def jqfilter(infile, outfile, jq_filter="."):
    with open(infile, "r") as f:
        # Use jq to filter the JSON
        result = subprocess.run(
            ["jq", jq_filter], stdin=f, capture_output=True, text=True
        )
    # Check output
    if result.returncode == 0:
        modified_json = result.stdout
        with open(outfile, "w") as f:
            f.write(modified_json)
    else:
        print("Error:", result.stderr)


def unzipper(infile, outdir):
    with open(infile, "r") as f:
        subprocess.run(
            ["unzip", infile, '-d', outdir], stdin=f, capture_output=True, text=False
        )

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
    count = int(obj[0]['count'])
    return count


# Explicitly define what gets imported when using `from models import *`
__all__ = ['get_file', 'downloader', 'jqfilter', 'unzipper', 'getDatasetRowCount']