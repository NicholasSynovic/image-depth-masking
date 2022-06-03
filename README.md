# Image Depth Masking

> An experiment to generate masks for images based off of estimated MiDaS depth

## Table of Contents

- [Image Depth Masking](#image-depth-masking)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
    - [Note: Repository Undergoing Maintence](#note-repository-undergoing-maintence)
    - [Supported Datasets](#supported-datasets)
  - [How to Install](#how-to-install)
    - [Packaged Code](#packaged-code)
    - [From Source](#from-source)
  - [How to Run](#how-to-run)
    - [COCO specific entrypoints](#coco-specific-entrypoints)
  - [How to Develop/ Extend](#how-to-develop-extend)

## About

This is a [cam2]() project.

It is a usage of the [MiDaS]() depth estimation models to generate masks for
images in order to reduce the computational intensity of computer vision (CV)
tasks.

### Note: Repository Undergoing Maintence

This repository was inherited from [Emmanual Amobi ()]().

I'm currently undergoing a refactoring effort to package and document his work.
Please parden the mess.

### Supported Datasets

Currently, the following datasets are supported:

- [COCO 2014]()
- [COCO 2017]()

The following datasets are planned to be incorporated:

- [MOT]()
- [CIFAR 10]()
- [CIFAR 100]()
- [Google Open Images Dataset]()

## How to Install

### Packaged Code

I release Python 3.10.4+ packages of this project here on GitHub.

Get the latest version [here]() and install using `pip`.

### From Source

This project uses [`poetry`]() as its build tool. You **will need it** to
package this project.

1. `git clone https://github.com/NicholasSynovic/image-depth-masking.git`
2. `cd image-depth-masking`
3. `poetry update`
4. `poetry build`
5. `poetry install dist/*.whl` or `poetry install dist/*.tar.gz`

## How to Run

To download datasets for testing, run the specific `download{}.bash` script
where `{}` is the dataset name. You will need [`parallel`]() installed to use
these scripts.

**WARNING**: This will download the *entire* dataset of your choosing which
could take up several gigabytes.

To manually download datasets, URLs are provided in the `{}URLS.txt` files where
`{}` is the dataset name.

Depending on the dataset that you are using, there are different entrypoints to
this tool.

### COCO specific entrypoints

- `idm`
  - ```
    usage: Find masks for images from the COCO Dataset,

    options:
    -a COCO_ANNOTATIONS_FILE, --coco-annotations-file COCO_ANNOTATIONS_FILE
                            A COCO annotations file in JSON format
    -d DEPTH_LEVEL, --depth-level DEPTH_LEVEL
                            The starting depth level to mask at. This should be between 0 and 1. This value is decremented by the
                            depth level decline value until the threshold is met. DEFAULT: 0.9
    -h, --help            show this help message and exit
    -i COCO_IMAGE_FOLDER, --coco-image-folder COCO_IMAGE_FOLDER
                            A path pointing to a folder of images from either the 2014 or 2017 COCO dataset.
    -l DEPTH_LEVEL_DECLINE, --depth-level-decline DEPTH_LEVEL_DECLINE
                            Set value that reduces the depth-level should a mask not be found at that level. This should be between
                            0 and 1. DEFAULT: 0.1
    -m MODEL, --model MODEL
                            A MiDaS compatible model. Supported arguements are: 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'. DEFAULT:
                            'MiDaS_small'.
    -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                            A directory to store masked images. DEFAULT: ./output.
    -s STEPPER, --stepper STEPPER
                            A stepper to step through the image folder. Helps reduce the number of images to be analyzed. DEFAULT:
                            1
    -t THRESHOLD, --threshold THRESHOLD
                            Threshold that must be met for the mask to be valid. This should be between 0 and 1. DEFAULT: 0.9
    -v, --version         Print version of the tool

    Tool created by Nicholas M. Synovic <nicholas.synovic@gmail.com,Emmanual Amobi <amobisomto@gmail.com>.
    ```

## How to Develop/ Extend

1. `git clone https://github.com/NicholasSynovic/image-depth-masking.git`
2. `cd image-depth-masking`
3. `poetry update`

Running these steps will ensure that you have all of the dependencies
installed and configured within a virtual environment that
[`poetry`]() created.

The [`clean.bash`](clean.bash) script is useful for cleaning your code and
generating reports.
