#!/bin/bash

directory="../downloads/COCO_2014"

mkdir -p ../downloads/$directory

wget -q -O $directory/annotations.zip --progress=bar --show-progress -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget -q -O $directory/validation.zip --progress=bar --show-progress -c http://images.cocodataset.org/zips/val2014.zip

wget -q -O $directory/training.zip --progress=bar --show-progress -c http://images.cocodataset.org/zips/train2014.zip
