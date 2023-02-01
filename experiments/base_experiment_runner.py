import numpy as np

from experiments.simple_mnist_model import SimpleMnistModel


class BaseExperimentRunner:
    def __init__(self, config, num_nodes, dataset="mnist"):
        self.num_nodes = num_nodes
        self.config = config
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.steps_per_epoch = config["steps_per_epoch"]
        self.lr = config["lr"]

        if dataset == "mnist":
            from tensorflow.keras.datasets import mnist

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # elif dataset == "cifar10":
        #     from tensorflow.keras.datasets import cifar10
        #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # ***currently works only for mnist***
    def create_models(self):
        return [SimpleMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]

    def random_split(self):
        num_partitions = self.num_nodes
        image_size = self.x_train.shape[1]
        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        # shuffle data then partition
        num_train = x_train.shape[0]
        indices = np.random.permutation(num_train)
        x_train = x_train[indices]
        y_train = self.y_train[indices]

        partitioned_x_train = np.array_split(x_train, num_partitions)
        partitioned_y_train = np.array_split(y_train, num_partitions)

        return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    # def skewed_split(self):
    #     num_partitions = self.num_nodes
    #     image_size = self.x_train.shape[1]
    #     x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
    #     x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
    #     x_train = x_train.astype(np.float32) / 255
    #     x_test = x_test.astype(np.float32) / 255

    #     # shuffle data then partition
    #     num_train = x_train.shape[0]
    #     indices = np.random.permutation(num_train)
    #     x_train = x_train[indices]
    #     y_train = self.y_train[indices]

    #     partitioned_x_train = np.array_split(x_train, num_partitions)
    #     partitioned_y_train = np.array_split(y_train, num_partitions)

    #     return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    def create_partitioned_datasets(self):
        num_partitions = self.num_nodes

        image_size = self.x_train.shape[1]
        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        (
            partitioned_x_train,
            partitioned_y_train,
        ) = self.split_training_data_into_paritions(
            x_train, self.y_train, num_partitions=num_partitions
        )
        return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    def get_train_dataloader_for_node(self, node_idx: int):
        partition_idx = node_idx
        partitioned_x_train = self.partitioned_x_train
        partitioned_y_train = self.partitioned_y_train
        while True:
            for i in range(0, len(partitioned_x_train[partition_idx]), self.batch_size):
                yield partitioned_x_train[partition_idx][
                    i : i + self.batch_size
                ], partitioned_y_train[partition_idx][i : i + self.batch_size]

    # ***currently this only works for mnist***
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
