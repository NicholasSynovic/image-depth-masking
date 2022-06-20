#!/bin/bash

directory="../downloads/STEP_ICCV21"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/mot20DET.zip --progress=bar --show-progress -c https://motchallenge.net/data/step_images.zip
