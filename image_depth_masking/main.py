from argparse import Namespace

import numpy
from numpy import ndarray
from pandas import DataFrame, Series
from PIL import Image
from progress.bar import Bar

from image_depth_masking.utils.args import maskArgs
from image_depth_masking.utils.files import getImagesInFolder, loadJSONData
from image_depth_masking.utils.masking import findMask
from image_depth_masking.utils.models import loadMiDaS, runMiDaS


def main():
    """
    main is a wrapper program to estimate depth with MiDaS and mask COCO images based off of their estimate depth level.
    """
    args: Namespace = maskArgs()

    imageFolder: dict = getImagesInFolder(
        folderPath=args.coco_image_folder, stepper=args.stepper
    )
    annotations: DataFrame = loadJSONData(
        jsonFilePath=args.coco_annotations_file
    )

    imageIDs: list = annotations["image_id"].unique().tolist()

    midasData: tuple = loadMiDaS(modelType=args.model)

    with Bar("Generating masks for images...", max=len(imageIDs)) as bar:
        id: int
        for id in imageIDs:
            imagePath: str = imageFolder[id]["path"]
            imageFilename: str = imageFolder[id]["filename"]

            boundingBoxs: Series = annotations[annotations["image_id"] == id][
                "bbox"
            ]
            depth: ndarray = runMiDaS(
                imagePath=imagePath,
                midas=midasData[0],
                device=midasData[1],
                transform=midasData[2],
            )
            maskData: tuple[float, ndarray, float, float, float] = findMask(
                imagePath=imagePath,
                boundingBoxs=boundingBoxs,
                depth=depth,
                depthLevel=args.depth_level,
                threshold=args.threshold,
                depthLevelDecline=args.depth_level_decline,
            )

            depthLevel: float = maskData[0]
            mask: ndarray = maskData[1]

            img_ = Image.open(imagePath)
            img_arr = numpy.array(img_)

            # set pixel of mask:0 to black, leave the rest as original color
            try:
                img_arr[mask.astype(bool), :] = 0
            except IndexError:
                print(f"\n{imagePath}")

            Image.fromarray(img_arr).save(
                f"{args.output_directory}/masked_{imageFilename}"
            )

            bar.next()


if __name__ == "__main__":
    main()
