from argparse import ArgumentParser, Namespace

from cam2_imagedepthmaps.utils.version import version

name: str = "cam2 Image Depth Maps: {}"
authors: list = [
    "Emmanual Amobi <amobisomto@gmail.com>",
    "Nicholas M. Synovic <nicholas.synovic@gmail.com",
]
versionString: str = name.format(version())


def maskArgs() -> Namespace:
    parser = ArgumentParser(
        prog=name.format("COCO Mask Finder"),
        usage="Find masks for images from the COCO Dataset",
        description="",
        epilog=f"Tool created by {','.join(authors)}",
    )
    parser.add_argument(
        "-i",
        "--image-folder",
        help="image folder name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-g",
        "--ground-truth-folder",
        help="ground truth folder name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="Print version of the tool",
        action="version",
        version=versionString,
    )
    return parser.parse_args()
