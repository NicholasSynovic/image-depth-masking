#!/bin/bash

directory="../downloads/COCO_2017"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/panopticAnnotations.zip --progress=bar --show-progress -c http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

wget -q -O $directory/stuffAnnotations.zip --progress=bar --show-progress -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

wget -q -O $directory/annotations.zip --progress=bar --show-progress -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

wget -q -O $directory/validation.zip --progress=bar --show-progress -chttp://images.cocodataset.org/zips/val2017.zip

wget -q -O $directory/traning.zip --progress=bar --show-progress -c http://images.cocodataset.org/zips/train2017.zip

unzip $directory/*.zip -d $directory

rm $directory/*.zip
