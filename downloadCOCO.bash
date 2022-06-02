#!/bin/bash

root=$PWD

# Download COCO datasets
mkdir COCO
mkdir COCO/2014
mkdir COCO/2017

# 2014
cd COCO/2014
cat ../../cocoURLS_2014.txt | parallel --bar "wget {}"
cd $root

# 2017
cd COCO/2017
cat ../../cocoURLS_2017.txt | parallel --bar "wget {}"
cd $root
