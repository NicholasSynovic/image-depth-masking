#!/bin/bash

# Download and unzip COCO test and evaluation images and annotations
mkdir COCO


# Download test and train evaluation images
wget -O COCO/train2014.zip http://images.cocodataset.org/zips/train2014.zip # 83K/13GB
wget -O COCO/evaluation2014.zip http://images.cocodataset.org/zips/val2014.zip # 41K/6GB

wget -O COCO/annotations2014.zip http://images.cocodataset.org/annotations/annotations_trainval2014.zip # 241MB

# Unzip files
cd COCO && ls | parallel --bar -j 3 "unzip {}"
