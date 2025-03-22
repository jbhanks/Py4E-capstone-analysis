from dataclasses import dataclass, field
from sqlalchemy.orm._orm_constructors import synonym
from sqlalchemy import Table
import re
from typing import Optional, List, Dict



# def Definition():
#     return {
#         'code': None,
#         'definition': None,
#         'etc': None,
#     }


@dataclass
class ColCustomization:
    """Class to hold customizations for a column"""

    short_name: str
    new_name: Optional[str] = None
    dtype: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    # definitions: dict = field(default_factory=Definition)
    drop: bool = False
    is_category: bool = False
    is_fk: bool = False
    orm: Optional[Table] = None

# Dataclass to hold information about an NYC Open Data dataset, to filled with metadata provided by the API
@dataclass
class DatasetInfo:
    """Class to hold dataset metadata"""
    name: str = None  # The name of the dataset
    short_name: str = None  # My shortened name for the dataset
    format: str = field(default="json")  # File format of the data_url
    id: str = field(init=False)  # the string ID used by NYC Open Data. In most cases, this is enough to construct all needed URLs.
    main_url: str = None  # The url for the dataset's main page
    metadata_url: str = field(init=False, default=None)  # The download URL for the dataset
    data_url: str = field(default=None)  # The download URL for the dataset
    dataset_path: str = None
    data_dict_path: str = None
    standard: bool = (True)  # Set to False if either the dataset is not hosted on NYC Open Data or follows a format different from the standard)
    geodata: bool = False
    metadata: dict = field(default_factory=dict)
    column_metadata: dict = field(default_factory=dict)
    cardinality_ratios: dict = field(default_factory=dict)
    data_dict_url: str = None  # The URL for the dataset's data dictionary
    other_files: List[str] = None  # A list of other files associated with the dataset
    attribution: str = (None)
    createdAt: str = None  # The date the dataset was created by NYC Open Data
    description: str = None  # NYC Open Data's description of the dataset
    provenance: str = None  # NYC Open Data's provenance for the dataset
    publicationDate: str = None  # Publication date of the dataset
    rowsUpdatedAt: str = (None)  # The date the dataset was last updated at the time of metadata retrieval
    col_types: dict = field(default_factory=dict)  # A dictionary of the columns in the dataset and their data types
    col_customizations: Dict[str, ColCustomization] = field(default_factory=dict)

    def __post_init__(self):
        self.id = re.sub(
            r".*/(.*)/about_data", r"\1", self.main_url
        )  # Extract the ID from the main URL
        self.metadata_url = f"https://data.cityofnewyork.us/api/views/{self.id}.json"
        if self.data_url is None:  # Only construct the `data_url` if it's not provided
            self.data_url = (
                f"https://data.cityofnewyork.us/api/views/{self.id}/rows.json?accessType=DOWNLOAD"
                if self.standard
                else None
            )


__all__ = ["DatasetInfo", "ColCustomization"]
