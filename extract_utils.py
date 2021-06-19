import collections
from typing import OrderedDict
import torch
import os
import sys
import math
import time
import torchvision
import random
from PIL import Image

# constants
# CUB_ROOT_DIR = "G://projects/pythonProjects/data/CUB_200_2011"
CUB_ROOT_DIR = "/home/computer/WBH/bmvc/CUB_200_2011"

IMAGE_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/images")
IMAGE_CLASS_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/classes.txt")
IMAGE_LABEL_DIR = os.path.join(
    CUB_ROOT_DIR, "CUB_200_2011/image_class_labels.txt"
)
IMAGE_ID_DIR_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/images.txt")
IMAGE_TRAIN_TEST_SPLIT_DIR = os.path.join(
    CUB_ROOT_DIR, "CUB_200_2011/train_test_split.txt"
)
PARTS_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/parts/parts.txt")
PARTS_LOC_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/parts/part_locs.txt")
BOUNDING_BOX_DIR = os.path.join(CUB_ROOT_DIR, "CUB_200_2011/bounding_boxes.txt")

CONCEPT_ROOT_DIR = os.path.join(CUB_ROOT_DIR, "concepts/")
RANDOM_CONCEPT_ROOT_DIR = os.path.join(CUB_ROOT_DIR, "random_concepts/")

SENSITIVITY_SAVE_DIR = os.path.join(CUB_ROOT_DIR, "sensitivity/")

VERBOSE = False


def read_file_to_dic(dic, dir, verbose=False, val_type=str):
    with open(dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            key, val = line.split(" ", 1)
            dic[key] = val_type(val)
            if verbose:
                print("%s: %s" % (key, dic[key]))


def extract(CONCEPT_ROOT_DIR):
    total_images = 200  # len(image_id_dir)

    for key, _ in parts.items():
        path = os.path.join(CONCEPT_ROOT_DIR, key)
        if not os.path.isdir(path):
            os.makedirs(path)

    cnt = 0
    timer_gap = 20
    t1 = time.time()
    for id, image_dir in image_id_dir.items():
        if train_test_split[id] == 0:
            continue
        with Image.open(os.path.join(IMAGE_DIR, image_dir)) as im:

            for key, val in parts_loc[id].items():
                shift = 128
                scale = 1024
                x, y = val
                region = (
                    bounding_box[id][0],
                    bounding_box[id][1],
                    bounding_box[id][0] + bounding_box[id][2],
                    bounding_box[id][1] + bounding_box[id][3],
                )
                crop = im.crop(region)
                scaled_y = math.floor(
                    scale / bounding_box[id][2] * bounding_box[id][3]
                )
                crop = crop.resize(
                    size=(scale, scaled_y), resample=Image.LANCZOS
                )
                x = x - bounding_box[id][0]
                y = y - bounding_box[id][1]
                x = math.floor(scale / bounding_box[id][2] * x)
                y = math.floor(scaled_y / bounding_box[id][3] * y)
                region = (
                    max(x - shift, 0),
                    max(y - shift, 0),
                    min(x + shift, scale),
                    min(y + shift, scaled_y),
                )
                crop = crop.crop(region)
                file_name = id + "." + im.format
                save_path = os.path.join(CONCEPT_ROOT_DIR, key, file_name)
                crop.save(save_path)
            cnt += 1
            if cnt % timer_gap == 0:
                t2 = time.time()
                left_time = ((total_images - cnt) / timer_gap) * (t2 - t1)
                hours = math.floor(left_time / 3600)
                mins = math.floor((left_time % 3600) / 60)
                secs = math.floor(left_time % 60)
                print(
                    "Processed %d images ETA: %d hours %.2d mins %.2d secs"
                    % (cnt, hours, mins, secs)
                )
                if cnt == total_images:
                    break
                t1 = time.time()


def extract_random(RANDOM_CONCEPT_ROOT_DIR):
    total_images = 200  # len(image_id_dir)
    random_exp_num = 500
    random_exp_dir_prefix = "random500_"
    for i in range(random_exp_num):
        path = os.path.join(
            RANDOM_CONCEPT_ROOT_DIR, random_exp_dir_prefix + str(i)
        )
        if not os.path.isdir(path):
            os.makedirs(path)

    cnt = 0
    random_exp_idx = 0
    timer_gap = 200
    path_per_img = 10
    t1 = time.time()
    for id, image_dir in image_id_dir.items():
        if train_test_split[id] == 0:
            continue
        with Image.open(os.path.join(IMAGE_DIR, image_dir)) as im:
            shift = 128
            scale = 1024
            scaled_y = math.floor(scale / im.size[0] * im.size[1])
            crop = im.resize((scale, scaled_y), resample=Image.LANCZOS)
            for j in range(path_per_img):
                x = random.randint(shift // 2, scale - shift // 2)
                y = random.randint(shift // 2, scaled_y - shift // 2)
                region = (
                    max(x - shift, 0),
                    max(y - shift, 0),
                    min(x + shift, scale),
                    min(y + shift, scaled_y),
                )
                crop_patch = crop.crop(region)
                file_name = id + "-" + str(j) + "." + im.format
                save_path = os.path.join(
                    RANDOM_CONCEPT_ROOT_DIR,
                    random_exp_dir_prefix + str(random_exp_idx),
                    file_name,
                )
                crop_patch.save(save_path)
                cnt += 1
                if cnt % 200 == 0:
                    random_exp_idx += 1
                if cnt % timer_gap == 0:
                    t2 = time.time()
                    left_time = (
                        (total_images * path_per_img - cnt) / timer_gap
                    ) * (t2 - t1)
                    hours = math.floor(left_time / 3600)
                    mins = math.floor((left_time % 3600) / 60)
                    secs = math.floor(left_time % 60)
                    print(
                        "Processed %d pathces ETA: %d hours %.2d mins %.2d secs"
                        % (cnt, hours, mins, secs)
                    )
                    t1 = time.time()
            if cnt >= total_images * path_per_img:
                break


# create sensitivity dir

if not os.path.exists(SENSITIVITY_SAVE_DIR):
    os.makedirs(SENSITIVITY_SAVE_DIR)

# get image_class
image_class = {}
read_file_to_dic(image_class, IMAGE_CLASS_DIR)

# get image_label
image_label = {}
read_file_to_dic(image_label, IMAGE_LABEL_DIR, val_type=int)
image_label = collections.OrderedDict(
    sorted(image_label.items(), key=lambda t: int(t[0]))
)

image_label_name = image_label.copy()
for key, val in image_label_name.items():
    image_label_name[key] = image_class[str(val)]

image_id_dir = {}
read_file_to_dic(image_id_dir, IMAGE_ID_DIR_DIR)
image_id_dir = collections.OrderedDict(
    sorted(image_id_dir.items(), key=lambda t: int(t[0]))
)

train_test_split = {}
read_file_to_dic(train_test_split, IMAGE_TRAIN_TEST_SPLIT_DIR, val_type=int)
train_test_split = collections.OrderedDict(
    sorted(train_test_split.items(), key=lambda t: int(t[0]))
)

parts = {}
read_file_to_dic(parts, PARTS_DIR)

parts_loc = {}
for key in image_id_dir.keys():
    parts_loc[key] = {}
with open(PARTS_LOC_DIR, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        id, part_id, x, y, visible = line.split()
        if int(visible) != 0:
            parts_loc[id][part_id] = (
                int(x.split(".")[0]),
                int(y.split(".")[0]),
            )
            if VERBOSE:
                print(
                    "id: %s part: %s x: %d y: %d visible: %s"
                    % (
                        id,
                        part_id,
                        parts_loc[id][part_id][0],
                        parts_loc[id][part_id][1],
                        visible,
                    )
                )
parts_loc = collections.OrderedDict(
    sorted(parts_loc.items(), key=lambda t: int(t[0]))
)

bounding_box = {}
with open(BOUNDING_BOX_DIR, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        id, x, y, w, h = line.split()
        bounding_box[id] = (
            int(x.split(".")[0]),
            int(y.split(".")[0]),
            int(w.split(".")[0]),
            int(h.split(".")[0]),
        )
        if VERBOSE:
            print(
                "id: %s, x: %d, y: %d, w: %d, h: %d"
                % (
                    id,
                    bounding_box[id][0],
                    bounding_box[id][1],
                    bounding_box[id][2],
                    bounding_box[id][3],
                )
            )
bounding_box = collections.OrderedDict(
    sorted(bounding_box.items(), key=lambda t: int(t[0]))
)

# print("total images: ", len(image_id_dir))

# print("Start concept extraction")
# extract(CONCEPT_ROOT_DIR)
# print("Start random concept extraction")
# extract_random(CONCEPT_ROOT_DIR)
