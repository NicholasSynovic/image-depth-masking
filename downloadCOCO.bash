#!/bin/bash

# Download and unzip COCO test and evaluation images and annotations
mkdir COCO


# Download test and train evaluation images
wget -O COCO/evaluation2017.zip http://images.cocodataset.org/zips/val2017.zip # 5K/1GB
# wget -O COCO/evaluation2014.zip http://images.cocodataset.org/zips/val2014.zip # 41K/6GB

wget -O COCO/panoptic2017.zip http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip # 821MB

# Unzip files
cd COCO && ls | parallel --bar -j 3 "unzip {}"
