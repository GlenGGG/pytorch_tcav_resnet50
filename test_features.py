import argparse
import os
import extract_utils
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        help="test train features",
        action="store_true",
        default=False,
    )
    group.add_argument(
        "--test", help="test test features", action="store_true", default=False
    )
    args = parser.parse_args()
    is_train = args.train
    scheme_str = "train" if is_train else "test"
    index = np.array(list(extract_utils.train_test_split.values())) == (
        1 if is_train else 0
    )
    print("index: ", index)
    size = sum(index == 1).item()
    print("{} size: ".format(scheme_str), size)
    features = np.memmap(
        "{}_features.npy".format(scheme_str),
        dtype=np.float32,
        mode="r",
        shape=(size, 15, 2048),
    )

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
    # parts_locs = parts_locs[index]
    parts_locs_rel = np.load("{}_parts_locs_rel.npy".format(scheme_str))

    for i in range(parts_locs_rel.shape[0]):
        for j in range(15):
            if parts_locs_rel[i, j, 2] == 0:
                # print(
                #     "i: {}, id: {}, part_loc: {}, features[i,j,:10]: {}".format(
                #         i, j + 1, parts_locs[i], features[i, j, :10]
                #     )
                # )
                assert sum(features[i, j, :] != 0).item() == 0, (
                    "i: {}, id: {}, part_loc: {}, features[i,j,:10]: {}"
                ).format(
                    i,
                    j + 1,
                    parts_locs_rel[i],
                    features[i, j, :10],
                )
            else:
                assert sum(features[i, j, :] != 0).item() > 0, (
                    "i: {}, id: {}, part_loc: {}, features[i,j,:10]: {}"
                ).format(
                    i,
                    j + 1,
                    parts_locs_rel[i],
                    features[i, j, :10],
                )
