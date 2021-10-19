import cv2
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import sys
# import os

def get_argparse():
    parser = ArgumentParser(
        prog="quick start with midas model",
        usage="This generates inverse depth map of an image using Midas model",
    )
    parser.add_argument(
        "-i",
        "--image",
        help="image for the model",
        type=str,
        # required=True,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="optional, select model to use: 0: DPT_Large, 1: DPT_Hybrid, 2: MiDaS_small, default -> 2: small",
        default=2,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Pass a folder of images",
        type=str,
        required=False,
    )
    return parser

def get_midas(model_type):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # pylint: disable=no-member
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform, device

def depth(img, midas, transform, device):
    img_tr = cv2.imread(img) # pylint: disable=no-member
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2RGB) # pylint: disable=no-member

    input_batch = transform(img_tr).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_tr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output


def get_folder_images(folder_name):
    """
    Gets images from folder 
    """
    image_exts = (".jpg", ".jpeg", ".png")
    images_ = [os.path.join(folder_name,_) for _ in os.listdir(folder_name) if _.lower().endswith(image_exts)]
    print(images_)




def main():
    models = ["DPT_Large","DPT_Hybrid", "MiDaS_small"]
    args = get_argparse().parse_args()
    get_folder_images(args.folder)
    sys.exit(0)
    filename = args.image
    output = "dept_map_" + models[args.model]+ "_" + filename 
    midas, transform, device = get_midas(models[args.model])
    plt.imshow(depth(filename, midas, transform, device))
    plt.savefig(output)

if __name__ == "__main__":
    main()
