#!/bin/bash

directory="../downloads/MOT_16"

mkdir -p ../downloads/$directory

wget -q -O $directory/mot16.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT16.zip
