## Install requirements
- pip install -r requirements.txt

## How to run
- for help on cli arguments: run `python main.py --help`

### To generate an image depth map
- put a folder of images into the root directory
- then run `python main.py -i <image_folder>`
- optional argument for model: you can specify the model(0,1,2) to use to run e.g: `python main.py -i <image_folder> -m 0`, use `--help` for more info
- this outputs the depth maps as csv files of numpy array to a folder `<${image_folder}/depth_maps>`

### To create image mask
- with the new depth maps folder generated, you can create an image mask
- run `python main.py -df <depth_map_folder>`
- optional, threshold value: this is a value range 0 - 1, to specify threshold for the mask. e.g: `python main.py -df <depth_map_folder> -t 0.5`, use `--help` for more info
- this outputs the image masks as csv files of numpy array to a folder `<${image_folder}/${threshold}_masks>`

### To apply the image mask
- with the new image mask folder generated, you can apply the mask to the image
- run `python main.py -i <image_folder> -mf <mask_folder> `
- this outputs a folder of images with the mask applied to a folder `<${image_folder}/applied_mask>`
