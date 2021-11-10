import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image

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



