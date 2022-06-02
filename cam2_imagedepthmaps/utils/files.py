"""
Code to handle the file system and file manipulations.
"""

from io import TextIOWrapper
import os.path
from pathlib import Path, PosixPath

import json

from pandas import DataFrame
import pandas


def getImagesInFolder(folderPath: str) -> tuple | bool:
    """
    getImagesInFolder returns a tuple containing all image paths and filenames.

    getImagesInFolder returns a tuple of lists containing all valid image paths and filenames with and without the file extension. A valid image path is only if the file ends in: `.jpg`, `.jpeg`, `.png`.

    A False `bool` is returned if the folder is invalid.

    :param folderPath: A folder path
    :type folderPath: str
    :return: A tuple of the valid image paths and filenames with and without the file extensions or a False `bool`
    :rtype: tuple | bool
    """
    if not os.path.isdir(folderPath):
        return False

    imagePaths: list = []
    filenames: list = []
    strippedFilenames: list = []
    extensions: tuple = (".jpg", ".jpeg", ".png")
    path: Path = Path(folderPath)

    obj: PosixPath
    for obj in path.iterdir():
        if obj.is_file() and obj.suffix in extensions:
            imagePaths.append(obj.absolute().__str__())
            filenames.append(obj.name)
            strippedFilenames.append(obj.with_suffix('').__str__())

    return (imagePaths, filenames, strippedFilenames)


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
