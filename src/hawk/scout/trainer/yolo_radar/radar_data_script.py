# SPDX-FileCopyrightText: 2022 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

# type: ignore

import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .yolov5_radar.radar_util import drawer, helper, loader

# steps for generating derivative dataset, with respec to size for each class.
# 1: find mean and standard deviation for a set of instances for a given class.
# set width of data to mean + stdev

# loop through all original images, get the azimuth center coordinate
# based on azimuth center coordinate and width, check if left and right side
# of width goes past 0 on the left or past 255 on the right.
# if width extends past 0 on the left, extract 0 to right side of width, then
# pad 0s to left up to width

# if width extends past 255 to the right, extract left side to 255, and pad 0s
# up to (width - 255)

# determine three section ranges, if total width % 3 == 0, 3 equal bins, if
# width % 3 == 1, ex 3, 4, 3, if width % 3 == 2, ex 4, 3, 4.
# set pixel ranges for each of the 3 sections, and take abs(), sum, then log.
# Once all data samples have been written, need to compute the mean and st dev
# across all samples and dimensions.  Rewrite the .npy files with normalized
# values.

ORIGINAL_IMAGES_DIR = Path()  # Set this
ORIGINAL_LABELS_DIR = Path()  # Set this
DESTINATION_IMAGES_DIR = None
DESTINATION_LABELS_DIR = None
DESTINATION_CANVAS_DIR = None
label_files = Path(ORIGINAL_LABELS_DIR).glob("*.pickle")
RAD_files = Path(ORIGINAL_IMAGES_DIR).glob("*.npy")
az_width_mean = 51
range_width_mean = 74
dopp_width_mean = 51
# The values above were determined by calculating the mean and st dev for each
# class and using such values for the largest class.  These three number
# represent mean + st dev for each class.


def visualize(RAD, RD_path) -> None:
    # This function is only used for generating new heat maps if wanting to
    # visualize the new created samples.
    RD = np.load(RD_path)
    RD = np.sum(RD, axis=2)
    RD = np.transpose(RD)
    RD_img = helper.norm2Image(RD)
    RD_img = RD_img[..., :3]

    prefix = ORIGINAL_IMAGES_DIR
    img_file = loader.imgfileFromRADfile(RAD, prefix)
    stereo_left_image = loader.readStereoLeft(img_file)
    fig, axes = drawer.prepareFigure(2, figsize=(7, 5))

    tile_name = RD_path.split("/")[-1].split(".")[0]
    new_label_name = DESTINATION_LABELS_DIR + tile_name + ".txt"
    gt_instances = loader.readRadarInstances(new_label_name)
    classes = list(range(6))
    colors = loader.randomColors(classes)

    drawer.clearAxes(axes)
    drawer.drawRadarBoxes(
        stereo_left_image,
        RD_img,
        gt_instances,
        classes,
        colors,
        axes,
    )

    tile_num = RD_path.split("/")[-1].split(".")[0]
    drawer.saveFigure(os.path.join(DESTINATION_CANVAS_DIR, f"{tile_num:.20s}.png"))


def slice_rad_file(
    rad_file,
    az_left,
    az_right,
    range_top,
    range_bottom,
    dopp_left,
    dopp_right,
):
    # This function takes a npy file path of the original RADDETR dataset npy
    # tensor along with the 3D extraction boundaries from the original npy
    # tensor.  It returns the final carved npy tensor of the shape (64, 256, 3)
    # which contains signal power for exactly one instance per new npy tensor.
    RAD = np.load(rad_file)
    RAD = np.transpose(RAD, (1, 0, 2))  # convert from RAD to ARD tensors
    if az_left < 0:
        az_left_boundary = 0
        az_pad_left = -az_left
    else:
        az_left_boundary = az_left
        az_pad_left = 0
    if az_right > 255:
        az_right_boundary = 255
        az_pad_right = az_right - 255
    else:
        az_right_boundary = az_right
        az_pad_right = 0

    if dopp_left < 0:
        dopp_left_boundary = 0
        dopp_pad_left = 0  ## orig: -dopp_left
    else:
        dopp_left_boundary = dopp_left
        dopp_pad_left = dopp_left  ## orig zero
    if dopp_right > 63:
        dopp_right_boundary = 64
        dopp_pad_right = 0  # orig: dopp_right - 63
    else:
        dopp_right_boundary = dopp_right
        dopp_pad_right = 64 - dopp_right  # orig: 0

    if range_top < 0:
        range_top_boundary = 0
        range_pad_top = 0  # orig: -range_top
    else:
        range_top_boundary = range_top
        range_pad_top = range_top  # orig: 0
    if range_bottom > 255:
        range_bottom_boundary = 256
        range_pad_bottom = 0  # orig: range_bottom - 255
    else:
        range_bottom_boundary = range_bottom
        range_pad_bottom = 256 - range_bottom  # orig: 0

    RD_unpadded = RAD[
        az_left_boundary:az_right_boundary,
        range_top_boundary:range_bottom_boundary,
        dopp_left_boundary:dopp_right_boundary,
    ]
    RD_padded = np.pad(
        RD_unpadded,
        (
            (az_pad_left, az_pad_right),
            (range_pad_top, range_pad_bottom),
            (dopp_pad_left, dopp_pad_right),
        ),
    )
    if az_width_mean % 3 == 0:
        left_1, right_1 = 0, int(az_width_mean / 3)
        left_2, right_2 = int(az_width_mean / 3), int(2 * az_width_mean / 3)
        left_3, right_3 = int(2 * az_width_mean / 3), az_width_mean
    elif az_width_mean % 3 == 2:
        left_1, right_1 = 0, int(az_width_mean / 3) + 1
        left_2, right_2 = int(az_width_mean / 3) + 1, int(2 * az_width_mean / 3)
        left_3, right_3 = int(2 * az_width_mean / 3), az_width_mean
    else:
        left_1, right_1 = 0, int(az_width_mean / 3)
        left_2, right_2 = int(az_width_mean / 3), int(2 * az_width_mean / 3) + 1
        left_3, right_3 = int(2 * az_width_mean / 3) + 1, az_width_mean

    ## absolute value and square
    RD = np.abs(RD_padded)
    RD = pow(RD, 2)

    ## separate the three sections
    RD_1 = RD[left_1:right_1, :, :]
    RD_2 = RD[left_2:right_2, :, :]
    RD_3 = RD[left_3:right_3, :, :]

    ## sum each section to a single channel
    RD_1 = np.sum(RD_1, axis=0)
    RD_2 = np.sum(RD_2, axis=0)
    RD_3 = np.sum(RD_3, axis=0)

    ## log
    RD_1 = np.log10(RD_1 + 1)
    RD_2 = np.log10(RD_2 + 1)
    RD_3 = np.log10(RD_3 + 1)

    ## stack the three channels to a 3 channel tensor
    RD_3_channel = np.stack([RD_1, RD_2, RD_3], axis=0)

    ## return the new RAD file
    return np.transpose(RD_3_channel, (2, 1, 0))


def get_label_data(label):
    # This function takes a single label file (pickle file) path from the
    # original RADDET dataset, loops through the object instances, and returns
    # a list for each on the return line: obj_list is the list of class names,
    # az_center_list is the list of all azimuth centers, and so forth.
    obj_list = []
    az_center_list, range_center_list, dopp_center_list = [], [], []
    az_width_list, range_width_list, dopp_width_list = [], [], []

    with open(label, "rb") as f:
        data = pickle.load(f)

    for obj_num, obj in enumerate(data["classes"]):
        obj_list.append(obj)
        az_center_list.append(data["boxes"][obj_num][1])
        az_width_list.append(data["boxes"][obj_num][3])
        range_center_list.append(data["boxes"][obj_num][0])
        range_width_list.append(data["boxes"][obj_num][3])
        dopp_center_list.append(data["boxes"][obj_num][2])
        dopp_width_list.append(data["boxes"][obj_num][5])
    return (
        obj_list,
        az_center_list,
        az_width_list,
        range_center_list,
        range_width_list,
        dopp_center_list,
        dopp_width_list,
    )


class_labels = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
}
global_class_count = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
az_running_width = {
    "car": [],
    "motorcycle": [],
    "bicycle": [],
    "person": [],
    "truck": [],
    "bus": [],
}
range_running_width = {
    "car": [],
    "motorcycle": [],
    "bicycle": [],
    "person": [],
    "truck": [],
    "bus": [],
}
dopp_running_width = {
    "car": [],
    "motorcycle": [],
    "bicycle": [],
    "person": [],
    "truck": [],
    "bus": [],
}
az_avg_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
range_avg_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
dopp_avg_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
az_sd_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
range_sd_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
dopp_sd_width = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "person": 0,
    "truck": 0,
    "bus": 0,
}
new_sample_count = 0
for label_file in tqdm(sorted(label_files)):
    part, label_base = label_file.split("/")[-2:]
    image_name = os.path.join(
        ORIGINAL_IMAGES_DIR,
        part + "/" + label_base.split(".")[0] + ".npy",
    )
    class_count = {
        "car": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "person": 0,
        "truck": 0,
        "bus": 0,
    }
    (
        obj_list,
        az_cent,
        az_width,
        range_cent,
        range_width,
        dopp_cent,
        dopp_width,
    ) = get_label_data(label_file)
    for i, obj in enumerate(obj_list):
        class_count[obj] += 1
        global_class_count[obj] += 1

        # these lines only needed for computing mean and stdev for all channels.
        # az_running_width[obj].append(az_width[i])
        # range_running_width[obj].append(range_width[i])
        # dopp_running_width[obj].append(dopp_width[i])

        if (az_cent[i] + az_width_mean / 2) % 1 == 0:
            az_left = int(az_cent[i] - az_width_mean / 2)
            az_right = int(az_cent[i] + az_width_mean / 2)
        else:
            az_left = int(az_cent[i] + 0.5 - az_width_mean / 2)
            az_right = int(az_cent[i] + 0.5 + az_width_mean / 2)

        if (range_cent[i] + range_width_mean / 2) % 1 == 0:
            range_top = int(range_cent[i] - range_width_mean / 2)
            range_bottom = int(range_cent[i] + range_width_mean / 2)
        else:
            range_top = int(range_cent[i] + 0.5 - range_width_mean / 2)
            range_bottom = int(range_cent[i] + 0.5 + range_width_mean / 2)

        if (dopp_cent[i] + dopp_width_mean / 2) % 1 == 0:
            dopp_left = int(dopp_cent[i] - dopp_width_mean / 2)
            dopp_right = int(dopp_cent[i] + dopp_width_mean / 2)
        else:
            dopp_left = int(dopp_cent[i] + 0.5 - dopp_width_mean / 2)
            dopp_right = int(dopp_cent[i] + 0.5 + dopp_width_mean / 2)

        RAD_3_ch = slice_rad_file(
            image_name,
            int(az_left),
            int(az_right),
            int(range_top),
            int(range_bottom),
            int(dopp_left),
            int(dopp_right),
        )
        new_RAD_file_name = (
            label_base.split(".")[0] + "_" + obj + "_" + str(class_count[obj]) + ".npy"
        )
        total_new_file_name = os.path.join(DESTINATION_IMAGES_DIR, new_RAD_file_name)

        new_label_name = (
            label_base.split(".")[0] + "_" + obj + "_" + str(class_count[obj]) + ".txt"
        )
        total_new_label_file_name = os.path.join(DESTINATION_LABELS_DIR, new_label_name)
        label_line = " ".join(
            [
                str(class_labels[obj]),
                str(int(dopp_cent[i])),
                str(int(range_cent[i])),
                str(int(dopp_width[i])),
                str(int(range_width[i])),
            ],
        )
        new_sample_count += 1
        if new_sample_count % 100 == 0:
            print(f"Sample Count: {new_sample_count}")
        with open(total_new_label_file_name, "w") as f:
            f.write(label_line)

        with open(total_new_file_name, "wb") as f:
            np.save(f, RAD_3_ch)

        ## visualize 1 sample at a time: RD map and stereo image
        # visualize(image_name, total_new_file_name)


# The code block below is only used to compute the radar transform means and
# standard deviations for each channel.  (The torch transforms when training
# and inferencing).
"""
for obj in az_avg_width:
    az_avg_width[obj] = np.mean(az_running_width[obj])
    az_sd_width[obj] = np.std(az_running_width[obj])
    range_avg_width[obj] = np.mean(range_running_width[obj])
    range_sd_width[obj] = np.std(range_running_width[obj])
    dopp_avg_width[obj] = np.mean(dopp_running_width[obj])
    dopp_sd_width[obj] = np.std(dopp_running_width[obj])
    print(
        f"{obj}, AZ avg: {az_avg_width[obj]}, AZ SD: {az_sd_width[obj]}, "
        f"RANGE avg: {range_avg_width[obj]}, RANGE sd width: {range_sd_width[obj]}, "
        f"Dopp avg width: {dopp_avg_width[obj]}, Dopp sd width: {dopp_sd_width[obj]}"
    )
"""
