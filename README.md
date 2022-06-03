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

Depending on the dataset that you are using, there are different entrypoints to
this tool.

### COCO specific entrypoints

`idm`

## How to Develop/ Extend
