import argparse
import PIL.Image
import numpy as np
from torchvision.models.resnet import Bottleneck
import cav as cav
import model as model
import tcav as tcav
from tcav import TCAV
import utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import pickle
import activation_generator as act_gen
import tensorflow as tf
import extract_utils
from multiprocessing import dummy as multiprocessing
from torchvision import transforms
from itertools import repeat
import matplotlib.pyplot as plt


def get_rel_parts_loc(image_size, ori_image_size, ori_x, ori_y):
    width, height = ori_image_size
    transform_scale = image_size / (min(width, height))
    resized_x = ori_x * transform_scale
    resized_y = ori_y * transform_scale
    x = resized_x - (transform_scale * width - image_size) // 2
    y = resized_y - (transform_scale * height - image_size) // 2
    return x, y


def show_image_and_parts(imgs, parts_locs, shape):
    for i in range(imgs.shape[0]):
        img = imgs[i].permute(1, 2, 0).view(shape[0], shape[1], -1)
        plt.imshow(img)
        print(parts_locs.shape)
        plt.scatter(
            [x for x in parts_locs[i, :, 0]],
            [y for y in parts_locs[i, :, 1]],
        )
        plt.show()


def load_image_from_file(filename, shape, parts_loc, bounding_box):
    """Given a filename, try to open the file. If failed, return None.
    Args:
      filename: location of the image file
      shape: the shape of the image file to be scaled
      part_loc: the location of the parts/concepts in this image
    Returns:
      the image (tensor, shape of (3,image_size,image_size)) if succeeds,
        None if fails.
      the parts' relative location to the transformed image
    """
    image_size = shape[0]
    transform = transforms.Compose(
        [
            # transforms.Resize(int(image_size / 0.875)),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    if not tf.io.gfile.exists(filename):
        tf.compat.v1.logging.error("Cannot find file: {}".format(filename))
        return None

    image = PIL.Image.open(tf.io.gfile.GFile(filename, "rb")).convert("RGB")
    # print("before transform's shape: ", image.size)
    img = transform(image)
    # print("after transform's shape: ", img.shape)

    scale = 1024 / bounding_box[2]
    parts_rel_loc = np.zeros((15, 3))
    for id, part_loc in parts_loc.items():

        parts_rel_loc[int(id) - 1, :2] = get_rel_parts_loc(
            image_size, image.size, part_loc[0], part_loc[1]
        )
        if (
            parts_rel_loc[int(id) - 1, 0] < 0
            or parts_rel_loc[int(id) - 1, 1] < 0
            or parts_rel_loc[int(id) - 1, 0] >= image_size
            or parts_rel_loc[int(id) - 1, 1] >= image_size
        ):
            parts_rel_loc[int(id) - 1, :] = 0
        else:
            parts_rel_loc[int(id) - 1, 2] = 1

    if not (len(img.shape) == 3 and img.shape[0] == 3):
        tf.compat.v1.logging.error(
            "wrong shape: {}, shape is : {}".format(filename, img.shape)
        )
        return None
    else:
        return img, parts_rel_loc, scale


def load_images_from_files(
    filenames,
    parts_locs,
    bounding_boxes,
    max_imgs=40,
    shape=(299, 299),
    run_parallel=False,
    num_workers=10,
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
    image_size = shape[0]
    imgs = torch.empty(
        (min(len(filenames), max_imgs), 3, image_size, image_size)
    )
    scales = []
    parts_locs_np = []

    cnt = 0
    if run_parallel:
        pool = multiprocessing.Pool(num_workers)
        # def load_image_from_file(filename, shape, parts_loc, bounding_box):
        img_pool = pool.starmap(
            load_image_from_file,
            zip(
                filenames[:max_imgs],
                repeat(shape),
                parts_locs[:max_imgs],
                bounding_boxes[:max_imgs],
            ),
        )
        # print(img_pool)
        for img, parts_loc, scale in img_pool:
            if img is not None:
                # imgs.append(img)
                img = img.view(1, 3, shape[0], shape[1])
                imgs[cnt] = img
                cnt += 1
                parts_locs_np.append(parts_loc)
                scales.append(scale)
        if cnt != min(len(filenames), max_imgs):
            print(imgs.shape, " ", min(len(filenames), max_imgs))
            raise ValueError("Failed to extract all images.")
    else:
        for filename in filenames:
            img, parts_loc, scale = load_image_from_file(
                filename, shape, parts_locs[cnt], bounding_boxes[cnt]
            )
            if img is not None:
                img = img.view(1, 3, image_size, image_size)
                imgs[cnt] = img
                parts_locs_np.append(parts_loc)
                scales.append(scale)
                cnt += 1
            if cnt >= max_imgs:
                break
        if cnt != min(len(filenames), max_imgs):
            print(imgs.shape, " ", min(len(filenames), max_imgs))
            raise ValueError("Failed to extract all images.")

    return imgs, np.array(parts_locs_np), np.array(scale)


def cal_features(
    imgs,
    parts_locs,
    bottleneck,
    modelwrapper: model.CUBResNet50Wrapper,
    image_size,
):
    features = modelwrapper.run_examples(imgs, bottleneck, True)
    # print("features.shape: ", features.shape)
    feature_len = features.shape[-1]
    feature_parts = np.zeros((features.shape[0], 15, features.shape[1]))
    assert parts_locs[:, :, :2].any() < 224
    feature_parts_loc = np.floor(
        parts_locs[:, :, :2] * (feature_len / image_size)
    ).astype(int)
    for i in range(feature_parts.shape[0]):
        for j in range(feature_parts.shape[1]):
            if parts_locs[i, j, 2] < 0.9:
                continue
            # Note, the feature's shape is N, C, H, W
            if feature_parts_loc[i, j, 1] > 6 or feature_parts_loc[i, j, 0] > 6:
                print("out of bound", parts_locs[i, j, :])
                print(
                    "i: ",
                    i,
                    "h: ",
                    feature_parts_loc[i, j, 1],
                    "w: ",
                    feature_parts_loc[i, j, 0],
                )
            feature_parts[i, j, :] = features[
                i, :, feature_parts_loc[i, j, 1], feature_parts_loc[i, j, 0]
            ]
    return feature_parts


def extract_features(mymodel, is_train=True):
    scheme_str = "train" if is_train else "test"
    index = np.array(list(extract_utils.train_test_split.values())) == (
        1 if is_train else 0
    )
    print("{}_index: ".format(scheme_str), index)
    size = sum(index == 1).item()
    print("{} size: ".format(scheme_str), size)
    features = np.memmap(
        "{}_features.npy".format(scheme_str),
        dtype=np.float32,
        mode="w+",
        shape=(size, 15, 2048),
    )

    labels = (np.array(list(extract_utils.image_label.values())) - 1)[index]
    np.save("{}_labels.npy".format(scheme_str), labels)

    cnt = 0
    gap = 40

    _image_dirs = extract_utils.image_id_dir.values()
    image_dirs = [
        os.path.join(extract_utils.CUB_ROOT_DIR, extract_utils.IMAGE_DIR, d)
        for d in _image_dirs
    ]
    image_dirs = np.array(image_dirs)
    bounding_boxes = np.array(list(extract_utils.bounding_box.values()))
    parts_locs = np.array(list(extract_utils.parts_loc.values()))
    del _image_dirs

    image_dirs = image_dirs[index]
    bounding_boxes = bounding_boxes[index]
    parts_locs = parts_locs[index]
    parts_locs_rel = np.zeros((size, 15, 3))

    while cnt < size:
        end_idx = min(cnt + gap, size)
        imgs_slice, parts_locs_slice, _ = load_images_from_files(
            image_dirs[cnt:end_idx],
            parts_locs[cnt:end_idx],
            bounding_boxes[cnt:end_idx],
            max_imgs=gap,
            shape=(224, 224),
        )

        parts_locs_rel[cnt:end_idx] = parts_locs_slice
        # show_image_and_parts(imgs_slice, parts_locs_slice, (224, 224))
        # print(imgs_slice.shape)
        feature_parts = cal_features(
            imgs_slice, parts_locs_slice, feature_bottlenecks, mymodel, 224
        )
        features[cnt:end_idx, :, :] = feature_parts
        cnt += gap

    np.save("{}_parts_locs_rel.npy".format(scheme_str), parts_locs_rel)
    features.flush()


def extract_sensitivity(
    mymodel: model.CUBResNet50Wrapper,
    act_generator: act_gen.ImageActivationGenerator,
    bottleneck,
    concepts,
    alpha,
    random_exp_num,
    cav_dir,
    run_parallel=False,
):
    train_index = np.array(list(extract_utils.train_test_split.values())) == 1
    print("train_index: ", train_index)
    train_size = sum(train_index == 1).item()
    print("train size: ", train_size)

    # labels = (np.array(list(extract_utils.image_label.values())) - 1)[
    #     train_index
    # ]
    label_names = (np.array(list(extract_utils.image_label_name.values())))[
        train_index
    ]
    # print(label_names[:10])

    _image_dirs = extract_utils.image_id_dir.values()
    image_dirs = [
        os.path.join(extract_utils.CUB_ROOT_DIR, extract_utils.IMAGE_DIR, d)
        for d in _image_dirs
    ]
    image_dirs = np.array(image_dirs)
    del _image_dirs

    image_dirs = image_dirs[train_index]

    train_sensitivity = np.zeros((train_size, 15))

    cnt = 0
    gap = 40
    while cnt < train_size:
        end_idx = min(cnt + gap, train_size)
        imgs_slice = act_generator.load_images_from_files(
            image_dirs[cnt:end_idx],
            max_imgs=gap,
            shape=(224, 224),
            do_shuffle=False,
        )

        class_acts = mymodel.run_examples(imgs_slice, bottleneck, True)

        if run_parallel:
            pass
        else:
            for i in range(class_acts.shape[0]):
                for j in range(len(concepts)):
                    train_sensitivity[i, j] = TCAV.get_sensitivities(
                        mymodel,
                        class_acts[i : i + 1],
                        concepts[j],
                        label_names[cnt:end_idx][i].item(),
                        bottleneck,
                        alpha,
                        random_exp_num,
                        cav_dir,
                    )
        np.save("train_sensitivity.npy", train_sensitivity)


def validate(namespace, parser):
    # print([arg for arg in vars(namespace).values()])
    if not any(arg for arg in vars(namespace).values()):
        parser.print_help()
        parser.exit(2)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(20)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        help="Extract train features",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test",
        help="Extract test features",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--sensitivity",
        help="Extract training set sensitivity",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    validate(args, parser)

    working_dir = "./tcav_class_test"
    activation_dir = working_dir + "/activations/"
    cav_dir = working_dir + "/cavs/"
    dataset = "CUB"

    if dataset == "CUB":
        source_dir = extract_utils.CONCEPT_ROOT_DIR
        print(source_dir)
        # concepts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
        concepts = ["3", "4"]
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
    bottleneck = "layer3"
    # bottlenecks = ['layer4']

    feature_bottlenecks = "layer4"

    alpha = 0.01

    act_generator = act_gen.ImageActivationGenerator(
        mymodel, source_dir, activation_dir, max_examples=40
    )

    feature_bank_save_root = "feature_bank"
    train_feature_bank_save_dir = os.path.join(
        feature_bank_save_root, "train", "feature_bank.np"
    ).replace("\\", "/")
    test_feature_bank_save_dir = os.path.join(
        feature_bank_save_root, "test", "feature_bank.np"
    ).replace("\\", "/")

    if not os.path.exists(os.path.join(feature_bank_save_root, "train")):
        os.makedirs(os.path.join(feature_bank_save_root, "train"))
    if not os.path.exists(os.path.join(feature_bank_save_root, "test")):
        os.makedirs(os.path.join(feature_bank_save_root, "test"))

    if args.train:
        extract_features(mymodel)
    if args.test:
        extract_features(mymodel, False)
    if args.sensitivity:
        extract_sensitivity(
            mymodel, act_generator, bottleneck, concepts, alpha, 100, cav_dir
        )
