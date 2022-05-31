"""
Code to handle the file system and file manipulations.
"""

import os.path
from pathlib import Path, PosixPath


def getImagesInFolder(folderPath: str) -> tuple | bool:
    """
    getImagesInFolder returns a list containing all image paths.

    getImagesInFolder returns a list containing a all valid image paths. A valid image path is only if the file ends in: `.jpg`, `.jpeg`, `.png`.

    A False `bool` is returned if the folder is invalid.

    :param folderPath: A folder path
    :type folderPath: str
    :return: A list of the valid image paths or a False `bool`
    :rtype: list | bool
    """
    if not os.path.isdir(folderPath):
        return False

    imagePaths: list = []
    extensions: tuple = (".jpg", ".jpeg", ".png")
    path: Path = Path(folderPath)

    obj: PosixPath
    for obj in path.iterdir():
        if obj.is_file() and obj.suffix in extensions:
            imagePaths.append(obj.absolute())

    return imagePaths
