import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image
#TODO: Write proper docs for all functions 
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

    
## TODO: put 90% for percentage for groundtruth. i.e 90% of original 1s are blacked out(changed to 0s)

def find_mask(depth_array, img_file, bboxes, thresh=0.9): 
    im_ar = np.array(Image.open(img_file))
    shape_ = im_ar.shape[0:2]
    gt_arr = np.zeros(shape_) #groundtruth map array

    fill_gt_bbox(gt_arr,bboxes) # populate bounding box with 1s
    total_ones_gt = np.count_nonzero(gt_arr) # count 1s in groundtruth array
    
    # add to for loop
    found_mask = False
    counter = 9
    mask_arr = []
    while not found_mask:
        mask_arr = create_img_mask(depth_array, counter*0.1)
        output = np.logical_and(mask_arr,gt_arr)
        output_ones = np.count_nonzero(output)
        percentage_covered = (total_ones_gt - output_ones) / total_ones_gt

        if percentage_covered >= thresh:
            found_mask = True

    return mask_arr









    

    
