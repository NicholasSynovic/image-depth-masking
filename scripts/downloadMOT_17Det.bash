#!/bin/bash

directory="../downloads/MOT_17DET"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/mot17DET.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOT17Det.zip

unzip $directory/*.zip -d $directory

rm $directory/*.zip
