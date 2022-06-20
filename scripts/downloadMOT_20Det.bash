#!/bin/bash

directory="../downloads/MOT_20DET"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/mot20DET.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT20Det.zip
