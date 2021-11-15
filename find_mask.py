import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
from main import get_folder_images
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
        "-df",
        "--depth_folder",
        help="Depth map folder name",
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
        depth_level -= 1
    return depth_level, mask_arr

def parse_MOT_gt(gt_file):
    headers = ["frame","id","bb_left","bb_top","bb_width","bb_height"]
    df = pd.read_csv('ADL_batch1/gt.txt',delimiter=",", header=None, usecols=list(range(0,6)))
    df.columns = headers

    df_grouped = df.groupby('frame') #group by image frame 
    return df_grouped

def find_mask_on_MOT_images(image_folder,depth_folder, gt_folder):

    df_stats = pd.DataFrame(columns=("Image","Depth_level","Useful_pixels(%)"))
    _, images = get_folder_images(image_folder)
    depths_ = [_ for _ in os.listdir(depth_folder) if _.lower().endswith(".csv")] # get csv files
    gt_file = os.path.join(gt_folder,'gt.txt')

    # get root folder
    root_folder = image_folder.split('/')[0]

    output_path_masks = os.path.join(root_folder,"masks")
    if not os.path.exists(output_path_masks): os.makedirs(output_path_masks)

    images = sorted(images)
    depths_ = sorted(depths_)
    df_grouped = parse_MOT_gt(gt_file)

    for group,df_group in df_grouped:
        # print(group)
        image_ = images[group-1]
        image_ = os.path.join(image_folder,image_)

        depth_path = depths_[group-1]
        depth_path = os.path.join(depth_folder,depth_path)
        depth_arr = np.genfromtxt(depth_path,delimiter=',')
        bboxes = []
        
        for row_index, row in df_group.iterrows():
            bbox_points = [row["bb_left"],row["bb_top"],row["bb_width"],row["bb_height"]]
            # print("BBOX: ", bbox)
            # points = convert_bbox_to_slices(bbox)
            bboxes.append(bbox_points)

        print(bboxes)
        depth_level, mask = find_mask(depth_arr,image_,bboxes)

        print("depth: ",  depth_path, "image:" , image_ , "depth level: " , depth_level,"size: ", mask.size, "useless: ", np.count_nonzero(mask), )
        useful_pixels = (mask.size - np.count_nonzero(mask)) / mask.size
        print("______________________-")
        # useful_pixels = round(useful_pixels,2)
        useful_pixels = useful_pixels*100

        entry = {"Image": group, "Depth_level": depth_level, "Useful_pixels(%)": useful_pixels }
        df_stats = df_stats.append(entry, ignore_index=True)

        image_name = os.path.splitext(image_)[0]
        image_name = image_name.split('/')[-1] 
        output_mask = "depth_" + str(depth_level) + "_mask_" + image_name + ".csv"
        output_mask = os.path.join(output_path_masks, output_mask)
        np.savetxt(output_mask, mask, delimiter=",")

    stats = os.path.join(image_folder,'stats.csv')
    df_stats.to_csv(stats,index=False)


def main():
    args = get_argparse().parse_args()
    image_folder = args.image_folder
    depth_folder = args.depth_folder
    gt_folder = args.gt_folder

    find_mask_on_MOT_images(image_folder,depth_folder,gt_folder)

if __name__ == "__main__":
    main()
