import numpy
from numpy import ndarray
from pandas import Series
from PIL import Image


def sliceGroundTruthByBoundingBoxs(
    groundTruth: ndarray, boundingBoxs: Series
) -> ndarray:
    """
    sliceGroundTruthByBoundingBoxs changes values within the ground truth array to provide a mask of the bounding boxes.

    sliceGroundTruthByBoundingBoxs changes the values of the groundTruth `ndarray` from 0 to 1 if the values position determined by (row, column) is within a boundingBox.

    :param groundTruth: A `ndarray` of zeros in the shape of the image to be masked
    :type groundTruth: ndarray
    :param boundingBoxs: A `Series` of bounding boxes from the respective COCO annotations file
    :type boundingBoxs: Series
    :return: An updated `groundTruth` `ndarray`. The shape doesn't change
    :rtype: ndarray
    """
    boundingBox: list
    for boundingBox in boundingBoxs:
        rowIndex0: float = boundingBox[1]
        rowIndexN: int = int(rowIndex0 + boundingBox[3] + 1)
        rowIndex0: int = int(rowIndex0)

        columnIndex0: float = boundingBox[0]
        columnIndexN: int = int(columnIndex0 + boundingBox[2] + 1)
        columnIndex0: int = int(columnIndex0)

        # Assign 1 to values in rowIndex0:rowIndexN and in columnIndex0:columnIndexN
        groundTruth[rowIndex0:rowIndexN, columnIndex0:columnIndexN] = 1
    return groundTruth


def createMask(depth: ndarray, depthLevel: float) -> ndarray:
    """
    createMask converts a depth array into a boolean array.

    createMask converts a `depth` `ndarray` from MiDaS into a boolean `ndarray`. Values are converted to `True` only if the value is greater than the maximum value of the `ndarray` * the `depthLevel`.

    :param depth: A depth `ndarray` from MiDaS
    :type depth: ndarray
    :param depthLevel: A value used to calculate the threshold to assign a boolean value
    :type depthLevel: float
    :return: A boolean `ndarray`
    :rtype: ndarray
    """
    maximum: float = numpy.amax(depth)
    thresholdValue: float = depthLevel * maximum
    return numpy.where(depth >= thresholdValue, False, True)


def findMask(
    imagePath: str,
    boundingBoxs: Series,
    depth: ndarray,
    depthLevel: float = 0.9,
    threshold: float = 0.9,
    depthLevelDecline: float = 0.1,
) -> tuple[float, ndarray]:
    """
    findMask finds and creates a mask for an image using depth values from MiDaS.

    findMask finds and creates an `ndarray` meant to be used to mask an image where any value of 1 is the mask and any value of 0 is transparent. The mask is the same shape as the image. It is genereated by utilizng `depth` values from MiDas and `boundingBoxes` from COCO annotations. In the event that the percent size of the mask is less than the `threshold`, the mask is regenerated at a lower `depthLevel`. This `depthLevel` is one `depthLevelDecline` less than the base `depthLevel`.

    :param imagePath: The filepath of the image to be masked
    :type imagePath: str
    :param boundingBoxs: A Series of COCO `bbox` annotations
    :type boundingBoxs: Series
    :param depth: Depth `ndarray` from MiDaS
    :type depth: ndarray
    :param depthLevel: Used to calculate the threshold for creating a depth mask, defaults to 0.9
    :type depthLevel: float, optional
    :param threshold: Used to determine if the percent size of the mask is at or greater than a certain percentage, defaults to 0.9
    :type threshold: float, optional
    :param depthLevelDecline: The amount to decremeant the `depthLevel` should the percent size of the mask not meet the `threshold`, defaults to 0.1
    :type depthLevelDecline: float, optional
    :return: A tuple containing the `depthLevel` and the mask as an `ndarray`
    :rtype: tuple[float, ndarray]
    """
    image: ndarray = numpy.array(Image.open(imagePath))

    imageShape: tuple = image.shape[0:2]
    groundTruth: ndarray = numpy.zeros(imageShape)

    groundTruth = sliceGroundTruthByBoundingBoxs(
        groundTruth=groundTruth, boundingBoxs=boundingBoxs
    )

    countGroundTruthOnes: int = numpy.count_nonzero(groundTruth)

    maskArray: ndarray = []
    while True:
        maskArray: ndarray = createMask(depth=depth, depthLevel=depthLevel)
        checkMask = numpy.logical_and(maskArray, groundTruth)
        countCheckMaskOnes = numpy.count_nonzero(checkMask)

        percentMasked: float = (
            countGroundTruthOnes - countCheckMaskOnes
        ) / countGroundTruthOnes

        if percentMasked >= threshold:
            break

        depthLevel -= depthLevelDecline
    return (depthLevel, maskArray)
