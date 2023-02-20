import numpy as np

# from flwr_p2p.keras.example import MnistModelBuilder
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
        elif dataset == "cifar10":
            from tensorflow.keras.datasets import cifar10
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    # ***currently works only for mnist***
    def create_models(self):
        # return [MnistModelBuilder(lr=self.lr).run() for _ in range(self.num_nodes)]
        return [SimpleMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]

    def normalize_data(self, data):
        image_size = data.shape[1]
        reshaped_data = np.reshape(data, [-1, image_size, image_size, 1])
        normalized_data = reshaped_data.astype(np.float32) / 255
        return normalized_data

    def random_split(self):
        num_partitions = self.num_nodes
        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

        # shuffle data then partition
        num_train = x_train.shape[0]
        indices = np.random.permutation(num_train)
        x_train = x_train[indices]
        y_train = self.y_train[indices]

        partitioned_x_train = np.array_split(x_train, num_partitions)
        partitioned_y_train = np.array_split(y_train, num_partitions)

        return partitioned_x_train, partitioned_y_train, x_test, self.y_test

    def create_skewed_partition_split(self, skew_factor: float = 0.5):
        # only works for 2 nodes at the moment
        # returns a "skewed" partition of data
        # Ex: 0.8 means 80% of the data for one node is 0-4 while 20% is 5-9
        # and vice versa for the other node
        # Note: A skew factor 0f 0.5 would essentially be a random split,
        # and 1 would be like a normal partition

        # num_partitions = self.num_nodes
        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

        x_train_by_label = [[] for _ in range(10)]
        y_train_by_label = [[] for _ in range(10)]
        for i in range(len(self.y_train)):
            label = self.y_train[i]
            x_train_by_label[label].append(x_train[i])
            y_train_by_label[label].append(label)

        skewed_partitioned_x_train = [[], []]
        skewed_partitioned_y_train = [[], []]
        for i in range(10):
            num_samples = len(x_train_by_label[i])
            num_samples_for_node_1 = int(num_samples * skew_factor)
            skewed_partitioned_x_train[int(i / 5)].extend(
                x_train_by_label[i][:num_samples_for_node_1]
            )
            skewed_partitioned_y_train[int(i / 5)].extend(
                y_train_by_label[i][:num_samples_for_node_1]
            )
            skewed_partitioned_x_train[int((i / 5)) - 1].extend(
                x_train_by_label[i][num_samples_for_node_1:]
            )
            skewed_partitioned_y_train[int((i / 5)) - 1].extend(
                y_train_by_label[i][num_samples_for_node_1:]
            )

        # convert to numpy arrays
        skewed_partitioned_x_train[0] = np.asarray(skewed_partitioned_x_train[0])
        skewed_partitioned_x_train[1] = np.asarray(skewed_partitioned_x_train[1])
        skewed_partitioned_y_train[0] = np.asarray(skewed_partitioned_y_train[0])
        skewed_partitioned_y_train[1] = np.asarray(skewed_partitioned_y_train[1])

        # shuffle data
        for i in range(2):
            num_train = skewed_partitioned_x_train[i].shape[0]
            indices = np.random.permutation(num_train)
            skewed_partitioned_x_train[i] = skewed_partitioned_x_train[i][indices]
            skewed_partitioned_y_train[i] = skewed_partitioned_y_train[i][indices]

        return (
            skewed_partitioned_x_train,
            skewed_partitioned_y_train,
            x_test,
            self.y_test,
        )

    def create_partitioned_datasets(self):
        num_partitions = self.num_nodes

        x_train = self.normalize_data(self.x_train)
        x_test = self.normalize_data(self.x_test)

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

    # ***currently this only works for mnist*** and for num_nodes = 2, 10
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


if __name__ == "__main__":
    config = {
        "epochs": 256,
        "batch_size": 32,
        "steps_per_epoch": 8,
        "lr": 0.001,
        "num_nodes": 2,
    }
    base_exp = BaseExperimentRunner(config, num_nodes=2)

    base_exp.random_split()
    base_exp.create_skewed_partition_split()
