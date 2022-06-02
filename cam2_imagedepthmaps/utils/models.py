"""
Code to load in models from model hubs
"""

from typing import Any
import torch

def loadMiDaS(modelType: str, forceCPU: bool = True) ->  tuple:
    """
    loadMiDaS loads a MiDaS model type from the PyTorch model hub.

    loadMiDaS downloads and validates a intel-isl/MiDaS model type from PyTorch's model hub. This is then loaded onto the device along with the MiDaS transforms model. The MiDaS model type, device information, and transformation model are returned wrapped in a tuple in that order. Code taken from https://pytorch.org/hub/intelisl_midas_v2/.

    :param modelType: MiDaS compatible model type
    :type modelType: str
    :param forceCPU: Flag to force PyTorch to load the MiDaS model to the CPU of the computer, defaults to True
    :type forceCPU: bool, optional
    :return: A tuple containing the following in order: the MiDaS model, device information, and the transformation model
    :rtype: tuple
    """
    print(f"Loading and validating intel-isl/MiDaS {modelType} for PyTorch...")
    midas: Any = torch.hub.load("intel-isl/MiDaS", modelType, verbose=False)

    print(f"Loading and validating intel-isl/MiDaS transforms for PyTorch...")
    midasTransforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)

    device: Any
    if forceCPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    midas.to(device)

    if modelType == "DPT_Large" or modelType == "DPT_Hybrid":
        transform: Any = midasTransforms.dpt_transform
    else:
        transform: Any = midasTransforms.small_transform

    return (midas, device, transform)
