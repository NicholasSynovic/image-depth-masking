#!/bin/bash

directory="../downloads/MOT_17DET"

mkdir -p ../downloads/$directory

wget -q -O $directory/mot17DET.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT17Det.zip
