import json
import os
import time
from argparse import Namespace
from datetime import timedelta

import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame

from cam2_imagedepthmaps.utils.args import maskArgs
from cam2_imagedepthmaps.utils.files import getImagesInFolder
from helper_funcs import depth, get_folder_images, get_midas


def create_img_mask(depth_array, threshold):
    """
    takes an image depth_array, with threshold/cutoff value
    Params:
        depth_array: np array, of depth values
        threshold: threshold value, between 0 and 1

    Returns:
        np array of booleans

    """
    max_ = np.amax(depth_array)
    thresh_val = threshold * max_
    return np.where(depth_array >= thresh_val, False, True)


def apply_mask_single(image_file, mask_arr):
    img_ = Image.open(image_file)
    img_arr = np.array(img_)
    img_arr[mask_arr.astype(bool), :] = 0

    return img_arr


def make_gt_map(image, bboxes):
    return 0


# convert groundtruth bbox format '<bb_left>, <bb_top>, <bb_width>, <bb_height>' to proper index for array slicing
def convert_bbox_to_slices(bbox):
    top = bbox[1]
    bottom = bbox[1] + bbox[3] + 1
    left = bbox[0]
    right = bbox[0] + bbox[2] + 1
    # return top,bottom,left,right
    return int(top), int(bottom), int(left), int(right)


def fill_gt_bbox(gt_arr, bboxes):
    for bbox in bboxes:
        top, bottom, left, right = convert_bbox_to_slices(bbox)
        gt_arr[top:bottom, left:right] = 1  # fill rectangle with ones


def find_mask(depth_array, img_file, bboxes, thresh=0.9):
    im_ar = np.array(Image.open(img_file))
    shape_ = im_ar.shape[0:2]
    gt_arr = np.zeros(shape_)  # groundtruth map array

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


# a json annotations file in coco format, parse the file and get groundtruth bounding boxes
def parse_COCO_gt(annotations_file):
    f = open(annotations_file)
    data = json.load(f)
    f.close()

    annotations = data["annotations"]
    image_set = set(
        list(map(lambda x: x["image_id"], annotations))
    )  # save set of images for optimized search

    return image_set, annotations


def findMasks(imageFolder: str, groundTruthFolder: str):
    stats: list = []
    models: list = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

    folder: tuple = getImagesInFolder(folderPath=imageFolder)
    imagePaths: list = folder[0]
    filenames: list = folder[1]

    _, images = get_folder_images(imageFolder)

    images.sort()

    image_set, annotations = parse_COCO_gt(groundTruthFolder)
    # helper
    def remove_ext(file_):
        return file_.split(".")[0]

    filtered_images = [
        im for im in images if int(remove_ext(im)) in image_set
    ]  # ignore images without annotations

    def take_first(item):
        return item[0]

    images_boxes = sorted(
        list(map(lambda x: [x["image_id"], x["bbox"]], annotations)),
        key=take_first,
    )
    # start processing images
    boxes_index = 0
    end = len(images_boxes)
    percentage_done = 0  # for tracking progress
    for i in filtered_images:
        image_ = os.path.join(imageFolder, i)
        midas, transform, device = get_midas(models[2])
        depth_arr = depth(image_, midas, transform, device)

        bboxes = []
        cur_image_id = int(remove_ext(i))

        # get bounding boxes
        while cur_image_id == int(images_boxes[boxes_index][0]):
            left = images_boxes[boxes_index][1][0]
            top = images_boxes[boxes_index][1][1]
            width = images_boxes[boxes_index][1][2]
            height = images_boxes[boxes_index][1][3]
            bbox_points = [left, top, width, height]
            bboxes.append(bbox_points)
            boxes_index += 1
            if boxes_index == end:
                break

        depth_level, mask = find_mask(depth_arr, image_, bboxes)
        useful_pixels = (mask.size - np.count_nonzero(mask)) / mask.size
        useful_pixels = useful_pixels * 100
        useful_pixels = round(useful_pixels, 4)

        entry = {
            "Image": i,
            "Depth_level": depth_level,
            "Useful_pixels(%)": useful_pixels,
        }
        stats.append(entry)

        image_name = os.path.splitext(image_)[0]
        image_name = image_name.split("/")[-1]

        percentage_done += 1
        done = (percentage_done / len(filtered_images)) * 100
        done = round(done, 2)
        print("================================")
        print("PERCENTAGE OF IMAGES DONE: {}%".format(done))
        print("================================")
        # output_mask = "depth_" + str(depth_level) + "_mask_" + image_name + ".csv"
        # output_mask = os.path.join(output_path_masks, output_mask)
        # np.savetxt(output_mask, mask, delimiter=",")

    statsPath: str = os.path.join(imageFolder, "stats.csv")
    DataFrame(stats).to_csv(statsPath, index=False)


def main():
    args: Namespace = maskArgs()

    startTime: float = time.monotonic()
    findMasks(imageFolder=args.image_folder, groundTruthFolder=args.ground_truth_folder)
    endTime: float = time.monotonic()

    print(f"Total elapsed time: {timedelta(seconds=endTime - startTime)}")


if __name__ == "__main__":
    main()
