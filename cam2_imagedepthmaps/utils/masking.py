import numpy
from numpy import ndarray
from PIL import Image


# def findMask(depth: ndarray, imagePath: str, bboxes: list, threshold: float=0.9):
def findMask(imagePath: str):
    image: ndarray = numpy.array(Image.open(imagePath))

    imageShape: tuple = image.shape[0:2]
    groundTruth: ndarray = numpy.zeros(imageShape)

    fill_gt_bbox(gt_arr, bboxes)  # populate bounding box with 1s
    total_ones_gt = np.count_nonzero(gt_arr)  # count 1s in groundtruth array

    found_mask = False
    depth_level = 9  # depth_level for trying depths
    mask_arr = []
    while not found_mask:
        mask_arr = create_img_mask(depth_array, depth_level * 0.1)
        output = np.logical_and(mask_arr, gt_arr)
        output_ones = np.count_nonzero(output)

        percentage_covered = (total_ones_gt - output_ones) / total_ones_gt

        if percentage_covered >= thresh:
            found_mask = True
            break

        depth_level -= 1
    return depth_level, mask_arr


findMask(imagePath="COCO/2014/train2014/COCO_train2014_000000144812.jpg")
