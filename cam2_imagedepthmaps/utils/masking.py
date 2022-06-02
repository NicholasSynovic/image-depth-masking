import numpy
from numpy import ndarray
from PIL import Image
from pandas import Series

def sliceGroundTruthByBoundingBoxs(groundTruth: ndarray, boundingBoxs: Series)  ->  ndarray:
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

def createMask(depth: ndarray, depthLevel: float)   ->  ndarray:
    maximum: float = numpy.amax(depth)
    thresholdValue: float = depthLevel * maximum
    return numpy.where(depth >= thresholdValue, False, True)

def findMask(imagePath: str, boundingBoxs: Series, depth: ndarray, depthLevel:float=0.9, threshold: float=0.9, depthLevelDecline: float = 0.1):
    image: ndarray = numpy.array(Image.open(imagePath))

    imageShape: tuple = image.shape[0:2]
    groundTruth: ndarray = numpy.zeros(imageShape)

    groundTruth = sliceGroundTruthByBoundingBoxs(groundTruth=groundTruth, boundingBoxs=boundingBoxs)

    countGroundTruthOnes: int = numpy.count_nonzero(groundTruth)

    maskArray: ndarray = []
    while True:
        maskArray: ndarray = createMask(depth=depth, depthLevel=depthLevel)
        checkMask = numpy.logical_and(maskArray, groundTruth)
        countCheckMaskOnes = numpy.count_nonzero(checkMask)

        percentMasked: float = (countGroundTruthOnes - countCheckMaskOnes) / countGroundTruthOnes

        if percentMasked >= threshold:
            break

        depthLevel -= depthLevelDecline
    return depthLevel, maskArray
