import logging
import argparse
import os
import time
import numpy as np
from multiprocessing import dummy as multiprocessing
from itertools import repeat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(filename="log.txt", level=logging.INFO)


def cal_distance(
    train_feature,
    train_sensitivity,
    test_feature,
    train_parts_visible,
    test_parts_visible,
    sensitivity_strength=1,
    with_sensitivity=True,
    plain_features=False,
):
    # print("train_feature.shape: ",train_feature.shape)
    # train/test_feature's shape: 15x1024
    if not plain_features:
        assert train_feature.shape == (15, 2048)
        # train_sensitivity's shape: 15
        assert train_sensitivity.shape == (15,)
        # part_visible's shape: 15
        # print("train_parts_visible.shape: ",train_parts_visible.shape)
        assert train_parts_visible.shape == (15,)
        assert test_parts_visible.shape == (15,)

        # dis's shape: 15
        dis = np.sqrt(np.sum(np.square(train_feature - test_feature), axis=1))
        if with_sensitivity:
            # only count visible parts in both train and test
            train_sensitivity = sensitivity_strength * (
                train_sensitivity * train_parts_visible * test_parts_visible
            )
            # softmax
            train_sensitivity = np.exp(train_sensitivity) / (
                np.sum(np.exp(train_sensitivity))
            )

            dis = np.dot(dis, train_sensitivity)
        else:
            dis = np.sum(dis)
    else:
        assert train_feature.shape == (2048,)
        dis = np.sqrt(np.sum(np.square(train_feature - test_feature)))
    return dis


def test(
    train_features,
    train_sensitivities,
    train_label,
    test_features,
    test_label,
    train_parts_locs,
    test_parts_locs,
    plain_features=False,
    with_sensitivity=True,
    sensitivity_strength=1,
    run_parallel=True,
    num_workers=10,
):
    correct = 0
    time1 = time.time()
    count_gap = 50
    for i in range(test_features.shape[0]):
        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            dis_pool = pool.starmap(
                cal_distance,
                zip(
                    train_features[:],
                    train_sensitivities[:],
                    repeat(test_features[i]),
                    train_parts_locs[:, :, 2],
                    repeat(test_parts_locs[i, :, 2]),
                    repeat(sensitivity_strength),
                    repeat(with_sensitivity),
                    repeat(plain_features),
                ),
            )
            dis_pool = np.array(dis_pool)
            assert dis_pool.shape == (train_features.shape[0],)
            predict = train_label[np.argmin(dis_pool)]
        else:
            dis_pool = np.empty((train_features.shape[0]))
            for j in range(train_features.shape[0]):
                dis_pool[j] = cal_distance(
                    train_features[j],
                    train_sensitivities[j],
                    test_features[i],
                    train_parts_locs[j, :, 2],
                    test_parts_locs[i, :, 2],
                    sensitivity_strength,
                    with_sensitivity,
                    plain_features,
                )
            predict = train_label[np.argmin(dis_pool)]

        if predict == test_label[i]:
            correct += 1
        if (i + 1) % count_gap == 0:
            time2 = time.time()
            time_per_sample = (time2 - time1) / count_gap
            total_left_estimate = (
                test_features.shape[0] - i - 1
            ) * time_per_sample
            hours = int(total_left_estimate // 3600)
            mins = int((total_left_estimate % 3600) // 60)
            secs = int(total_left_estimate % 60)
            logging.info(
                (
                    "{}  ETA: {:2d} hours, {:2d} mins, {:2d} secs"
                    ", average time per sample: {:.2f} secs"
                    ", acc: {:.2f}"
                ).format(
                    time.asctime(time.localtime(time2)),
                    hours,
                    mins,
                    secs,
                    time_per_sample,
                    float(correct) / (i + 1),
                )
            )
            time1 = time.time()
    return float(correct) / test_features.shape[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plain",
        help="Extract plain features without part info, e.g., the direct output of feature network",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--without_sensitivity",
        help="Extract training set sensitivity",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--sensitivity_strength",
        help="Extract training set sensitivity",
        default=1,
        type=int
    )

    args = parser.parse_args()

    train_label = np.load("train_labels.npy", allow_pickle=True)
    test_label = np.load("test_labels.npy", allow_pickle=True)
    if args.plain:
        train_features = np.memmap(
            "train_features_plain.npy",
            mode="r",
            dtype=np.float32,
            shape=(train_label.shape[0], 2048)
        )
        test_features = np.memmap(
            "test_features_plain.npy",
            mode="r",
            dtype=np.float32,
            shape=(train_label.shape[0], 2048)
        )
    else:
        train_features = np.memmap(
            "train_features.npy",
            mode="r",
            dtype=np.float32,
            shape=(train_label.shape[0], 15, 2048),
        )
        test_features = np.memmap(
            "test_features.npy",
            mode="r",
            dtype=np.float32,
            shape=(test_label.shape[0], 15, 2048),
        )
    train_parts_locs = np.load("train_parts_locs_rel.npy", allow_pickle=True)
    test_parts_locs = np.load("test_parts_locs_rel.npy", allow_pickle=True)
    train_sensitivities = np.load("train_sensitivities.npy", allow_pickle=True)

    accuracy = test(
        train_features,
        train_sensitivities,
        train_label,
        test_features,
        test_label,
        train_parts_locs,
        test_parts_locs,
        plain_features=args.plain,
        with_sensitivity=not args.without_sensitivity,
        sensitivity_strength=args.sensitivity_strength,
    )

    logging.info("Accuracy: {:.2f}".format(accuracy))

# def test(
#     train_features,
#     train_sensitivities,
#     train_label,
#     test_features,
#     test_label,
#     run_parallel=True,
#     num_workers=10,
# ):
