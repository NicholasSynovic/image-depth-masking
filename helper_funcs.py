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

def get_midas(model_type):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # pylint: disable=no-member
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform, device

def depth(img, midas, transform, device):
    img_tr = cv2.imread(img) # pylint: disable=no-member
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2RGB) # pylint: disable=no-member

    input_batch = transform(img_tr).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_tr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output
