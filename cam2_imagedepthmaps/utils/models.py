"""
Code to load in models from model hubs
"""

from typing import Any
import torch
from torch import device

def loadMIDaS(modelType: str) ->  tuple:
    print(f"Loading and validating intel-isl/MiDaS {modelType} for PyTorch...")
    midas: Any = torch.hub.load("intel-isl/MiDaS", modelType, verbose=False)

    print(f"Loading and validating intel-isl/MiDaS transforms for PyTorch...")
    midasTransforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)

    device: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    if modelType == "DPT_Large" or modelType == "DPT_Hybrid":
        transform: Any = midasTransforms.dpt_transform
    else:
        transform: Any = midasTransforms.small_transform

    return (midas, device, transform)



loadMIDaS("MiDaS_small")
