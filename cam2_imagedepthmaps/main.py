from argparse import Namespace

from numpy import ndarray
from pandas import DataFrame, Series
from progress.bar import Bar

from cam2_imagedepthmaps.utils.args import maskArgs
from cam2_imagedepthmaps.utils.files import getImagesInFolder, loadJSONData
from cam2_imagedepthmaps.utils.models import loadMiDaS, runMiDaS
from cam2_imagedepthmaps.utils.masking import findMask

# def create_img_mask(depth_array, threshold):
#     """
#     takes an image depth_array, with threshold/cutoff value
#     Params:
#         depth_array: np array, of depth values
#         threshold: threshold value, between 0 and 1

#     Returns:
#         np array of booleans

#     """
#     max_ = np.amax(depth_array)
#     thresh_val = threshold * max_
#     return np.where(depth_array >= thresh_val, False, True)


# def apply_mask_single(image_file, mask_arr):
#     img_ = Image.open(image_file)
#     img_arr = np.array(img_)
#     img_arr[mask_arr.astype(bool), :] = 0

#     return img_arr


# def make_gt_map(image, bboxes):
#     return 0


# # convert groundtruth bbox format '<bb_left>, <bb_top>, <bb_width>, <bb_height>' to proper index for array slicing
# def convert_bbox_to_slices(bbox):
#     top = bbox[1]
#     bottom = bbox[1] + bbox[3] + 1
#     left = bbox[0]
#     right = bbox[0] + bbox[2] + 1
#     # return top,bottom,left,right
#     return int(top), int(bottom), int(left), int(right)


# def fill_gt_bbox(gt_arr, bboxes):
#     for bbox in bboxes:
#         top, bottom, left, right = convert_bbox_to_slices(bbox)
#         gt_arr[top:bottom, left:right] = 1  # fill rectangle with ones


# def find_mask(depth_array, img_file, bboxes, thresh=0.9):
#     im_ar = np.array(Image.open(img_file))
#     shape_ = im_ar.shape[0:2]
#     gt_arr = np.zeros(shape_)  # groundtruth map array

#     fill_gt_bbox(gt_arr, bboxes)  # populate bounding box with 1s
#     total_ones_gt = np.count_nonzero(gt_arr)  # count 1s in groundtruth array

#     found_mask = False
#     depth_level = 9  # depth_level for trying depths
#     mask_arr = []
#     while not found_mask:
#         mask_arr = create_img_mask(depth_array, depth_level * 0.1)
#         output = np.logical_and(mask_arr, gt_arr)
#         output_ones = np.count_nonzero(output)

#         percentage_covered = (total_ones_gt - output_ones) / total_ones_gt

#         if percentage_covered >= thresh:
#             found_mask = True
#             break

#         depth_level -= 1
#     return depth_level, mask_arr


# def generateMaskMask(midas: Any, d)  ->  DataFrame

#     depth_level, mask = find_mask(depth_arr, image_, bboxes)
#     useful_pixels = (mask.size - np.count_nonzero(mask)) / mask.size
#     useful_pixels = useful_pixels * 100
#     useful_pixels = round(useful_pixels, 4)

#     entry = {
#         "Image": i,
#         "Depth_level": depth_level,
#         "Useful_pixels(%)": useful_pixels,
#     }
#     stats.append(entry)

#     image_name = os.path.splitext(image_)[0]
#     image_name = image_name.split("/")[-1]

#     percentage_done += 1
#     done = (percentage_done / len(filtered_images)) * 100
#     done = round(done, 2)
#     print("================================")
#     print("PERCENTAGE OF IMAGES DONE: {}%".format(done))
#     print("================================")
#     # output_mask = "depth_" + str(depth_level) + "_mask_" + image_name + ".csv"
#     # output_mask = os.path.join(output_path_masks, output_mask)
#     # np.savetxt(output_mask, mask, delimiter=",")

#     statsPath: str = os.path.join(imageFolder, "stats.csv")
#     DataFrame(stats).to_csv(statsPath, index=False)


def main():
    args: Namespace = maskArgs()

    imageFolder: dict = getImagesInFolder(folderPath=args.coco_image_folder)
    annotations: DataFrame = loadJSONData(
        jsonFilePath=args.coco_annotations_file
    )

    imageIDs: list = annotations["image_id"].unique().tolist()

    midasData: tuple = loadMiDaS(modelType=args.model)

    with Bar("Generating masks for images...", max=len(imageIDs)) as bar:
        id: int
        for id in imageIDs:
            imagePath: str = imageFolder[id]["path"]
            boundingBoxs: Series = annotations[annotations["image_id"] == id]["bbox"]
            depth: ndarray = runMiDaS(
                imagePath=imagePath,
                midas=midasData[0],
                device=midasData[1],
                transform=midasData[2],
            )
            findMask(imagePath=imagePath, boundingBoxs=boundingBoxs, depth=depth, depthLevel=0.9, threshold=0.9)

            bar.next()

        # findMask()

    # startTime: float = time.monotonic()
    # findMasks(imageFolder=args.image_folder, groundTruthFolder=args.ground_truth_folder)
    # endTime: float = time.monotonic()

    # print(f"Total elapsed time: {timedelta(seconds=endTime - startTime)}")


if __name__ == "__main__":
    main()
