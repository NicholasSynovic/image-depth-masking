import cv2
import torch
# import urllib.request
import matplotlib.pyplot as plt
# import os

models = ["DPT_Large","DPT_Hybrid", "MiDaS_small"]
model_type = models[2]
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = "cpu"
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def depth(img):
    img_tr = cv2.imread(img) # pylint: disable=no-member
    # print("img TR: ", img_tr)
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


def main():
    filename1,filename2 = 'artInstitutePic.jpg', 'nazam_wed.jpg'
    plt.imshow(depth(filename1))
    # _ = plt.figure(0)
    plt.figure(0) # get first image

    plt.imshow(depth(filename2))
    # _ = plt.figure(1) 
    plt.figure(1) # get second image

    plt.show()

if __name__ == "__main__":
    main()
