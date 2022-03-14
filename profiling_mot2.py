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
import timeit
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
    top = int(bbox[1])
    bottom = int(bbox[1] + bbox[3]+1)
    left = int(bbox[0])
    right = int(bbox[0] + bbox[2]+1)
    return top,bottom,left,right

def fill_gt_bbox(gt_arr, bboxes):
    for bbox in bboxes:
        top,bottom,left,right = convert_bbox_to_slices(bbox)
        gt_arr[top:bottom, left:right] = 1 # fill rectangle with ones

    
def find_mask(depth_array, img_file, bboxes, thresh=0.9): 
    im_ar = np.array(Image.open(img_file))
    shape_ = im_ar.shape[0:2]
    gt_arr = np.zeros(shape_) #groundtruth map array

    fill_gt_bbox(gt_arr,bboxes) # populate bounding box with 1s
    total_ones_gt = np.count_nonzero(gt_arr) # count 1s in groundtruth array
    
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

def parse_MOT_gt(gt_file):
    headers = {"frame":0,"id":1,"bb_left":2,"bb_top":3,"bb_width":4,"bb_height":5} # map headers to column index 
    data = np.loadtxt(gt_file, delimiter = ",",usecols=list(range(0,6)))
    image_ids = np.unique(data[:,0])

    return headers, data, image_ids

# check if current mask covers threshold percentage of groundtruth. Thresh -> (0.0 - 1.0)
def verify_gt(thresh, mask_arr, gt_arr, total_ones_gt): 
    output = np.logical_and(mask_arr,gt_arr)
    output_ones = np.count_nonzero(output)
    percentage_covered = (total_ones_gt - output_ones) / total_ones_gt
    return percentage_covered >= thresh

def find_mask_on_MOT_images(image_folder,gt_file):
    skipped_frames = []
    models = ["DPT_Large","DPT_Hybrid", "MiDaS_small"]

    df_stats = pd.DataFrame(columns=("Image","Depth_level","Useful_pixels(%)"))
    _, images = get_folder_images(image_folder)


    # get root folder
    root_folder = image_folder.split('/')[0]

    # output path for images with mask applied
    output_path = "applied_mask"
    output_path = os.path.join(root_folder,output_path)
    if not os.path.exists(output_path): os.makedirs(output_path)
    # =====================================

    images.sort()
    headers, data, image_ids = parse_MOT_gt(gt_file) # get groundtruth data

    row_index = 0

    cur_depth, cur_mask = None, None # keep track of current depth and current mask
    frame_skips = 0 # track number of frame skips(reused mask on multiple frames)
    end = np.shape(data)[0] # get row count
    for i in range(len(images)):
        image_ = os.path.join(image_folder,images[i])

        bboxes = []
        cur_image_id = image_ids[i]

        # get bounding boxes
        
        while cur_image_id == int(data[row_index][0]):
            row = data[i]
            left, top, width, height = row[headers["bb_left"]], row[headers["bb_top"]], row[headers["bb_width"]], row[headers["bb_height"]]
            bbox_points = [left, top, width, height]
            bboxes.append(bbox_points)
            row_index += 1
            if row_index == end:
                break

        # try reusing mask of previous image
        if i > 0:
            im_ar = np.array(Image.open(image_))
            shape_ = im_ar.shape[0:2]
            gt_arr = np.zeros(shape_) #groundtruth map array
            fill_gt_bbox(gt_arr,bboxes) # populate bounding box with 1s
            total_ones_gt = np.count_nonzero(gt_arr) # count 1s in groundtruth array
            if verify_gt(0.8, cur_mask, gt_arr, total_ones_gt):
                frame_skips += 1
            else:
                midas, transform, device = get_midas(models[2])
                depth_arr = depth(image_, midas, transform, device)
                cur_depth, cur_mask = find_mask(depth_arr,image_,bboxes, thresh=0.8)
                skipped_frames.append(frame_skips) # store skipped frames 
                frame_skips = 0 # reset frame counter
        else:
            midas, transform, device = get_midas(models[2])
            depth_arr = depth(image_, midas, transform, device)
            cur_depth, cur_mask = find_mask(depth_arr,image_,bboxes, thresh=0.8)
        # save images with mask applied
        # img_ = Image.open(image_)
        # img_arr = np.array(img_)
        # img_arr[mask.astype(bool), :] = 0 # set pixel of mask:0 to black, leave the rest as original color
        # image_no_ext = os.path.splitext(images[i])[0]
        # output =  "masked_" + image_no_ext + ".jpg"
        # output = os.path.join(output_path, output)
        # plt.imshow(img_arr)
        # plt.savefig(output)
        # =====================================


        # CALCULATING USEFUL PIXELS ======================
        useful_pixels = (cur_mask.size - np.count_nonzero(cur_mask)) / cur_mask.size
        # useful_pixels = round(useful_pixels,2)
        useful_pixels = useful_pixels*100

        entry = {"Image": cur_image_id, "Depth_level": cur_depth, "Useful_pixels(%)": useful_pixels }
        df_stats = df_stats.append(entry, ignore_index=True)

        image_name = os.path.splitext(image_)[0]
        image_name = image_name.split('/')[-1] 

    stats = os.path.join(image_folder,'stats.csv')
    df_stats.to_csv(stats,index=False)

    if frame_skips > 0:
        skipped_frames.append(frame_skips)

    if len(skipped_frames) > 0:
        file = open("skipped_frames.txt", "w+")
        content = str(skipped_frames)
        file.write(content)
        file.close()
    # skipped_output = np.array(skipped_frames)
    # np.savetxt("skipped_frames.txt", skipped_output)


def main():
    args = get_argparse().parse_args()
    image_folder = args.image_folder
    gt_folder = args.gt_folder

    start = timeit.default_timer()
    find_mask_on_MOT_images(image_folder,gt_folder)
    stop = timeit.default_timer()
    time_elapsed = "Time elapsed: " + str(stop-start)
    print("Time elapsed: ", stop - start )
    output_time = open("elapsed.txt","w")
    output_time.write(time_elapsed)
    output_time.close()

if __name__ == "__main__":
    main()
