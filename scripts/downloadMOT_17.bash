#!/bin/bash

directory="../downloads/MOT_17"

mkdir -p ../downloads/$directory

wget -q -O $directory/mot17.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT17.zip
