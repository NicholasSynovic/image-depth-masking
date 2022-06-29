#!/bin/bash

directory="../downloads/MOT_15"

mkdir -p ../downloads/$directory

wget -q -O $directory/mot15.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT15.zip
