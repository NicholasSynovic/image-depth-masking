"""
Code to handle command line arguments
"""

from argparse import ArgumentParser, HelpFormatter, Namespace
from operator import attrgetter

from cam2_imagedepthmaps.utils.version import version

name: str = "cam2 Image Depth Maps: {}"
authors: list = [
    "Emmanual Amobi <amobisomto@gmail.com>",
    "Nicholas M. Synovic <nicholas.synovic@gmail.com",
]
versionString: str = name.format(version())

class SortingHelpFormatter(HelpFormatter):
    """
    SortingHelpFormatter is a class to alphabetically sort command line arguements in the `help` (`-h`) output.

    This class sorts command line arguements in the `help` (`-h`) view of the command line. This class was written by Martijn Pieters on StackOverflow (https://stackoverflow.com/a/12269143).

    :param HelpFormatter: Formatter for generating usage messages and arguement help strings. See python.argparse.HelpFormatter for more information.
    :type HelpFormatter: class
    """

    def add_arguments(self, actions):
        """
        add_arguments adds arguments in alphabetical order.

        :param actions: an Iterable of Actions to be sorted alphabetically.
        :type actions: Iterable[Action]
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def versionArgument(parser: ArgumentParser) ->  None:
    """
    versionArgument adds the generic version text to ArgumentParsers.

    :param parser: The ArgumentParser to append the version argument to.
    :type parser: ArgumentParser
    """
    parser.add_argument(
        "-v",
        "--version",
        help="Print version of the tool",
        action="version",
        version=versionString,
    )


def maskArgs() -> Namespace:
    """
    maskArgs has all of the arguments for the cam2_imagedepthmaps.findMask_COCO.py program.

    :return: A Namespace containing the users input
    :rtype: Namespace
    """
    parser:ArgumentParser = ArgumentParser(
        prog=name.format("COCO Mask Finder"),
        usage="Find masks for images from the COCO Dataset",
        description="",
        epilog=f"Tool created by {','.join(authors)}",
        formatter_class=SortingHelpFormatter,
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
    versionArgument(parser=parser)

    return parser.parse_args()
