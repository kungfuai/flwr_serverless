import numpy as np


class BaseExperimentRunner:
    def __init__(self, config, dataset="mnist"):
        self.config = config
        if dataset == "mnist":
            from tensorflow.keras.datasets import mnist

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # elif dataset == "cifar10":
        #     from tensorflow.keras.datasets import cifar10
        #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # currently this only works for mnist***
    def split_training_data_into_paritions(
        self, x_train, y_train, num_partitions: int = 2
    ):
        # partion 1: classes 0-4
        # partion 2: classes 5-9
        # client 1 train on classes 0-4 only, and validated on 0-9
        # client 2 train on classes 5-9 only, and validated on 0-9
        # both clients will have low accuracy on 0-9 (below 0.6)
        # but when federated, the accuracy will be higher than 0.6
        classes = list(range(10))
        num_classes_per_partition = int(len(classes) / num_partitions)
        partitioned_classes = [
            classes[i : i + num_classes_per_partition]
            for i in range(0, len(classes), num_classes_per_partition)
        ]
        partitioned_x_train = []
        partitioned_y_train = []
        for partition in partitioned_classes:
            partitioned_x_train.append(x_train[np.isin(y_train, partition)])
            partitioned_y_train.append(y_train[np.isin(y_train, partition)])
        return partitioned_x_train, partitioned_y_train
