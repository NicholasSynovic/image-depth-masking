"""
Code to handle the file system and file manipulations.
"""

import os.path
from pathlib import Path, PosixPath


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

print(getImagesInFolder("."))
