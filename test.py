import time
import numpy as np
import torch
from multiprocessing import dummy as multiprocessing
from itertools import repeat


def cal_distance(train_feature, train_sensitivity, test_feature):
    # train/test_feature's shape: 15x2048
    # train_sensitivity's shape: 15

    # dis's shape: 15
    dis = np.sqrt(np.sum(np.square(train_feature - test_feature), axis=1))
    dis = np.dot(dis, train_sensitivity)
    return dis


def test(
    train_features,
    train_sensitivities,
    train_label,
    test_features,
    test_label,
    run_parallel=True,
    num_workers=10,
):
    train_sensitivities = torch.softmax(
        torch.from_numpy(train_sensitivities), dim=1
    ).numpy()
    correct = 0
    time1 = time.time()
    count_gap = 5
    for i in range(test_features.shape[0]):
        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            dis_pool = pool.starmap(
                cal_distance,
                zip(
                    train_features,
                    train_sensitivities,
                    repeat(test_features[i]),
                ),
            )
            dis_pool = np.array(dis_pool)
            assert dis_pool.shape == (train_features.shape[0],)
            predict = train_label[np.argmin(dis_pool)]
        else:
            dis_pool = np.empty((train_features.shape[0]))
            for j in range(train_features.shape[0]):
                dis_pool[j] = cal_distance(
                    train_features[j], train_sensitivities[j], test_features[i]
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
            print(
                (
                    "ETA: {:2d} hours, {:2d} mins, {:2d} secs"
                    ", average time per sample: {:d} secs"
                ).format(
                    hours,
                    mins,
                    secs,
                    time_per_sample,
                )
            )
    return float(correct) / test_features.shape[0]


if __name__ == "__main__":
    train_features = np.load("train_features.npy")
    test_features = np.load("test_features.npy")
    train_label = np.load("train_labels.npy")
    test_label = np.load("test_labels.npy")
    train_sensitivities = np.load("train_sensitivities.npy")

    accuracy = test(
        train_features,
        train_sensitivities,
        train_label,
        test_features,
        test_label,
    )

    print("Accuracy: {:.2f}".format(accuracy))

# def test(
#     train_features,
#     train_sensitivities,
#     train_label,
#     test_features,
#     test_label,
#     run_parallel=True,
#     num_workers=10,
# ):
