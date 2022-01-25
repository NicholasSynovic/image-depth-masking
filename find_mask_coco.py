import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
from helper_funcs import get_folder_images, get_midas, depth
import json
#TODO: Write proper docs for all functions 
def get_argparse():
    parser = ArgumentParser(
        prog="find good mask",
        usage="trys different depths until find good mask",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        help="image folder name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-gt",
        "--gt_folder",
        help="ground truth folder name",
        type=str,
        required=True,
    )
    return parser
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
    thresh_val = threshold*max_
    return np.where(depth_array >= thresh_val, False, True) 

def apply_mask_single(image_file,mask_arr):
    img_ = Image.open(image_file)
    img_arr = np.array(img_)
    img_arr[mask_arr.astype(bool), :] = 0

    return img_arr

def make_gt_map(image,bboxes):
    return 0


# convert groundtruth bbox format '<bb_left>, <bb_top>, <bb_width>, <bb_height>' to proper index for array slicing
def convert_bbox_to_slices(bbox):
    top = bbox[1]
    bottom = bbox[1] + bbox[3]+1
    left = bbox[0]
    right = bbox[0] + bbox[2]+1
    # return top,bottom,left,right
    return int(top),int(bottom),int(left),int(right)

def fill_gt_bbox(gt_arr, bboxes):
     for bbox in bboxes:
         top,bottom,left,right = convert_bbox_to_slices(bbox)
         gt_arr[top:bottom, left:right] = 1 # fill rectangle with ones

    
def find_mask(depth_array, img_file, bboxes, thresh=0.9): 
    im_ar = np.array(Image.open(img_file))
    shape_ = im_ar.shape[0:2]
    gt_arr = np.zeros(shape_) #groundtruth map array

    fill_gt_bbox(gt_arr,bboxes) # populate bounding box with 1s
    # print("BBOXES: ", bboxes)
    # print("IMAGE: ", img_file)
    total_ones_gt = np.count_nonzero(gt_arr) # count 1s in groundtruth array
    # print("UNIQUE: ", np.unique(gt_arr))
    # print("SIZEE: ", gt_arr.size)
    # print("TOTAL ONES: ", total_ones_gt)
    
    # add to for loop
    found_mask = False
    depth_level = 9 # depth_level for trying depths
    mask_arr = []
    while not found_mask:
        mask_arr = create_img_mask(depth_array, depth_level*0.1)
        output = np.logical_and(mask_arr,gt_arr)
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

    annotations = data['annotations']
    image_set = set(list(map(lambda x: x['image_id'], annotations))) # save set of images for optimized search

    # use for sorting
    # def take_first(item):
    #     return item[0]
    # images_boxes = sorted(list(map(lambda x: [x['image_id'], x['bbox']], annotations)),key=take_first)
    # headers = ["frame","bb_left","bb_top","bb_width","bb_height"]
    # df = pd.DataFrame(columns = headers)
    #
    # for i in images_boxes:
    #     frame = i[0]
    #     bb_left = i[1][0]
    #     bb_top = i[1][1]
    #     bb_width = i[1][2]
    #     bb_height = i[1][3]
    #     entry = {"frame": frame, "bb_left": bb_left, "bb_top": bb_top, "bb_width": bb_width, "bb_height": bb_height}
    #
    #     df = df.append(entry, ignore_index=True)
    # df_grouped = df.groupby('frame') #group by image frame 
    return image_set, annotations

def find_mask_on_COCO_images(image_folder, gt_file):
    models = ["DPT_Large","DPT_Hybrid", "MiDaS_small"]

    df_stats = pd.DataFrame(columns=("Image","Depth_level","Useful_pixels(%)"))
    _, images = get_folder_images(image_folder)

    # get root folder
    root_folder = image_folder.split('/')[0]

    # output_path_masks = os.path.join(root_folder,"masks")
    # if not os.path.exists(output_path_masks): os.makedirs(output_path_masks)

    # images = sorted(images)
    images.sort()

    image_set,annotations = parse_COCO_gt(gt_file)
    # helper
    def remove_ext(file_):
        return file_.split(".")[0]
    filtered_images = [im for im in images if int(remove_ext(im)) in image_set] # ignore images without annotations

    def take_first(item):
        return item[0]
    images_boxes = sorted(list(map(lambda x: [x['image_id'], x['bbox']], annotations)),key=take_first)
    # start processing images
    boxes_index = 0
    end = len(images_boxes)
    percentage_done = 0 # for tracking progress
    for i in filtered_images:
        image_ = os.path.join(image_folder,i)
        midas, transform, device = get_midas(models[2])
        depth_arr = depth(image_, midas, transform, device)

        bboxes = []
        cur_image_id = int(remove_ext(i))
        # get bounding boxes
        # print("IMAGE: ", image_)
        while cur_image_id == int(images_boxes[boxes_index][0]):
            left = images_boxes[boxes_index][1][0]
            top = images_boxes[boxes_index][1][1]
            width = images_boxes[boxes_index][1][2]
            height = images_boxes[boxes_index][1][3]
            bbox_points = [left,top,width,height]
            bboxes.append(bbox_points)
            # print(images_boxes[boxes_index], "InDEX: ", boxes_index)
            boxes_index += 1
            if boxes_index == end:
                break
        # print("++++++++++++++++++++++++++++++")
        depth_level, mask = find_mask(depth_arr,image_,bboxes)
        useful_pixels = (mask.size - np.count_nonzero(mask)) / mask.size
        # useful_pixels = round(useful_pixels,2)
        useful_pixels = useful_pixels*100
        useful_pixels = round(useful_pixels,4)

        entry = {"Image": i, "Depth_level": depth_level, "Useful_pixels(%)": useful_pixels }
        df_stats = df_stats.append(entry, ignore_index=True)

        image_name = os.path.splitext(image_)[0]
        image_name = image_name.split('/')[-1] 

        percentage_done += 1
        done = (percentage_done / len(filtered_images)) * 100
        done = round(done,2)
        print("================================")
        print("PERCENTAGE OF IMAGES DONE: {}%".format(done))
        print("================================")
        # output_mask = "depth_" + str(depth_level) + "_mask_" + image_name + ".csv"
        # output_mask = os.path.join(output_path_masks, output_mask)
        # np.savetxt(output_mask, mask, delimiter=",")

    #################################################################################################################
    # count = 0
    # for group,df_group in df_grouped:
    #     image_ = filtered_images[int(group-1)]
    #     image_ = os.path.join(image_folder,image_)
    #
    #     midas, transform, device = get_midas(models[2])
    #     depth_arr = depth(image_, midas, transform, device)
    #     bboxes = []
    #     
    #     print("IMAGE: ", image_)
    #     for row_index, row in df_group.iterrows():
    #
    #         print(row)
    #         bbox_points = [row["bb_left"],row["bb_top"],row["bb_width"],row["bb_height"]]
    #         # print("CURR IMAGE: ", image_)
    #         # print("BBOX POINTS: ", bbox_points)
    #         bboxes.append(bbox_points)
    #
    #     print("+++++++++++++++++++++++++++++")
    #     depth_level, mask = find_mask(depth_arr,image_,bboxes)
    #
    #     useful_pixels = (mask.size - np.count_nonzero(mask)) / mask.size
    #     # useful_pixels = round(useful_pixels,2)
    #     useful_pixels = useful_pixels*100
    #
    #     entry = {"Image": group, "Depth_level": depth_level, "Useful_pixels(%)": useful_pixels }
    #     df_stats = df_stats.append(entry, ignore_index=True)
    #
    #     image_name = os.path.splitext(image_)[0]
    #     image_name = image_name.split('/')[-1] 
    #     # output_mask = "depth_" + str(depth_level) + "_mask_" + image_name + ".csv"
    #     # output_mask = os.path.join(output_path_masks, output_mask)
    #     # np.savetxt(output_mask, mask, delimiter=",")

    stats = os.path.join(image_folder,'stats.csv')
    df_stats.to_csv(stats,index=False)

import time
from datetime import timedelta

def start_time_measure(message=None):
    if message:
        print(message)
    return time.monotonic()

def end_time_measure(start_time, print_prefix=None):
    end_time = time.monotonic()
    if print_prefix:
        print(print_prefix + str(timedelta(seconds=end_time - start_time)))
    return end_time
def main():
    total_start_time = start_time_measure('Generating output...')    
    # Do something
    args = get_argparse().parse_args()
    image_folder = args.image_folder
    # depth_folder = args.depth_folder
    gt_folder = args.gt_folder
    find_mask_on_COCO_images(image_folder,gt_folder)

    end_time_measure(total_start_time, 'Total time elapsed:')

if __name__ == "__main__":
    main()
