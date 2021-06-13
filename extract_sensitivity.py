import PIL.Image
import cv2
import numpy as np
import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import pickle
import activation_generator as act_gen
import tensorflow as tf
import extract_utils
from torchvision import transforms

image_size = 224
transform = transforms.Compose(
    [
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)


def load_image_from_file(filename, shape, parts_loc, bounding_box):
    """Given a filename, try to open the file. If failed, return None.
    Args:
      filename: location of the image file
      shape: the shape of the image file to be scaled
      part_loc: the location of the parts/concepts in this image
    Returns:
      the image (tensor, shape of (3,image_size,image_size)) if succeeds, None if fails.
      the parts' relative location to the transformed image
    """
    if not tf.io.gfile.exists(filename):
        tf.compat.v1.logging.error("Cannot find file: {}".format(filename))
        return None
    # try:
    # ensure image has no transparency channel
    image = PIL.Image.open(tf.io.gfile.GFile(filename, "rb")).convert("RGB")
    width, height = image.size
    img = transform(image)

    scale = 1024 / bounding_box[2]
    parts_rel_loc = np.zeros((15, 3))
    cnt = 0
    for _, part_loc in parts_loc:
        parts_rel_loc[cnt, 0] = part_loc[0]
        parts_rel_loc[cnt, 1] = part_loc[1]
        cnt += 1

    if not (len(img.shape) == 3 and img.shape[2] == 3):
        return None
    else:
        return img, parts_rel_loc, scale


def load_images_from_files(
    filenames,
    parts_locs,
    bounding_boxs,
    max_imgs=40,
    shape=(299, 299),
):
    """Return image arrays from filenames.
    Args:
      filenames: locations of image files.
      max_imgs: maximum number of images from filenames.
      do_shuffle: before getting max_imgs files, shuffle the names or not
      run_parallel: get images in parallel or not
      shape: desired shape of the image
      num_workers: number of workers in parallelization.
    Returns:
      imgs torch tensor of shape (n, 3, 224, 224)
    """
    imgs = torch.empty((0, 3, image_size, image_size))
    scales = []
    parts_locs_np = []
    filenames = filenames[:]

    cnt = 0
    for filename in filenames:
        img, parts_loc, scale = load_image_from_file(
            filename, shape, parts_locs[cnt], bounding_boxs[cnt]
        )
        cnt += 1
        if img is not None:
            img = img.view(1, 3, image_size, image_size)
            imgs = torch.cat([imgs, img], dim=0)
            parts_locs_np.append(parts_loc)
            scales.append(scale)
        if imgs.shape[0] <= 1:
            raise ValueError(
                "You must have more than 1 image in each class to run TCAV."
            )
        elif imgs.shape[1] >= max_imgs:
            break

    return imgs, np.array(parts_locs_np), np.array(scale)

def cal_features(imgs, parts_locs,bottleneck, modelwrapper:model.CUBResNet50Wrapper):
    features = modelwrapper.run_examples(imgs,bottleneck,True)
    print(features.shape)

working_dir = "./tcav_class_test"
activation_dir = working_dir + "/activations/"
cav_dir = working_dir + "/cavs/"
dataset = "CUB"

if dataset == "CUB":
    source_dir = extract_utils.CONCEPT_ROOT_DIR
    print(source_dir)
    # concepts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    concepts = ["1", "2"]
    LABEL_PATH = "./cub_200_2011_labels.txt"
    target_exclude = ["001.Black_footed_Albatross"]
    targets = tf.io.gfile.GFile(LABEL_PATH).read().splitlines()
    mymodel = model.CUBResNet50Wrapper(LABEL_PATH, "./82.12_best_model.tar")
else:
    source_dir = "./image_net_subsets"
    concepts = ["dotted", "striped"]
    LABEL_PATH = "./imagenet_comp_graph_label_strings.txt"
    target_exclude = ["001.Black_footed_Albatross"]
    targets = ["zebra"]
    # mymodel = model.InceptionV3Wrapper(LABEL_PATH)
    mymodel = model.ResNet50Wrapper(LABEL_PATH)

# bottlenecks = ['Mixed_5d', 'Conv2d_2a_3x3']
# bottlenecks = ['Conv2d_2a_3x3']
bottlenecks = ["layer1", "layer3"]
# bottlenecks = ['layer4']

feature_bottlenecks = ["layer1"]


alphas = [0.01]

act_generator = act_gen.ImageActivationGenerator(
    mymodel, source_dir, activation_dir, max_examples=40
)


labels = np.empty((0), dtype=int)
filenames = []

feature_bank_save_root = "feature_bank"
train_feature_bank_save_dir = os.path.join(
    feature_bank_save_root, "train", "feature_bank.np"
).replace("\\","/")
test_feature_bank_save_dir = os.path.join(
    feature_bank_save_root, "test", "feature_bank.np"
).replace("\\","/")

if not os.path.exists(os.path.join(feature_bank_save_root, "train")):
    os.makedirs(os.path.join(feature_bank_save_root, "train"))
if not os.path.exists(os.path.join(feature_bank_save_root, "test")):
    os.makedirs(os.path.join(feature_bank_save_root, "test"))

# train_features = np.memmap(
#     train_feature_bank_save_dir, dtype=np.float32, mode="w+", shape=(1, 15, 2048)
# )
# test_features = np.memmap(
#     train_feature_bank_save_dir, dtype=np.float32, mode="w+", shape=(1, 15, 2048)
# )
train_features = np.memmap(
    "train_features.np", dtype=np.float32, mode="w+", shape=(1, 15, 2048)
)
test_features = np.memmap(
    "test_features.np", dtype=np.float32, mode="w+", shape=(1, 15, 2048)
)

train_labels = np.empty((0))
test_labels = np.empty((0))

image_filenames = []

cnt = 0
pre_idx = 0
next_idx = 40
for id, dir in extract_utils.image_id_dir.items():
    dir = os.path.join(extract_utils.IMAGE_DIR,dir)
    image_filenames.append(dir)
    
    cnt+=1