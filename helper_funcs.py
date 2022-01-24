import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image

def get_folder_images(folder_name):
    """
    Gets images from folder 
    """
    image_exts = (".jpg", ".jpeg", ".png")
    images_path = [os.path.join(folder_name,_) for _ in os.listdir(folder_name) if _.lower().endswith(image_exts)]
    images_ = [_ for _ in os.listdir(folder_name) if _.lower().endswith(image_exts)]
    return images_path, images_
