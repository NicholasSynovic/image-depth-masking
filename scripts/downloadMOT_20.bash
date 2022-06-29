#!/bin/bash

directory="../downloads/MOT_20"

mkdir -p ../downloads/$directory

wget -q -O $directory/mot20.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT20.zip
