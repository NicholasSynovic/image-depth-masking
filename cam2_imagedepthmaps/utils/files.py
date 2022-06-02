"""
Code to handle the file system and file manipulations.
"""

from io import TextIOWrapper
import os.path
from pathlib import Path, PosixPath

import json

from pandas import DataFrame
import pandas


def getImagesInFolder(folderPath: str) -> dict | bool:
    """
    getImagesInFolder returns a dict containing all image paths and filenames.

    getImagesInFolder returns a dict of  all valid image paths and filenames. A valid image path is only if the file ends in: `.jpg`, `.jpeg`, `.png`.

    A False `bool` is returned if the folder is invalid.

    :param folderPath: A folder path
    :type folderPath: str
    :return: A dict of the valid image paths and filenames or a False `bool`
    :rtype: dict | bool
    """
    if not os.path.isdir(folderPath):
        return False

    data: dict = {}

    imagePaths: list = []
    filenames: list = []
    strippedFilenames: list = []
    extensions: tuple = (".jpg", ".jpeg", ".png")
    path: Path = Path(folderPath)

    obj: PosixPath
    for obj in path.iterdir():
        if obj.is_file() and obj.suffix in extensions:
            data[int(obj.with_suffix('').name.split("_")[-1])] = {
                "path": obj.absolute().__str__(),
                "filename": obj.name,
            }

    return data


def loadJSONData(jsonFilePath: str)   ->  DataFrame | bool:
    """
    loadJSONData loads a COCO annotations JSON file.

    loadJSONData loads a COCO annotations JSON file and returns a pandas DataFrame object if it has a "bbox" (bounding box) key. Else, a False boolean is returned

    :param jsonFilePath: filepath to the COCO annotations file.
    :type jsonFilePath: str
    :return: Either a DataFrame if it is a valid or usable COCO file, or a False boolean
    :rtype: DataFrame | bool
    """
    jsonFile: TextIOWrapper
    with open(jsonFilePath, "r") as jsonFile:
        jsonData: dict = json.load(jsonFile)
        jsonFile.close()

    df: DataFrame = DataFrame(jsonData["annotations"])

    if df.columns.__contains__("bbox"):
        return df
    return False
