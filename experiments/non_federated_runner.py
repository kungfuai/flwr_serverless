import numpy as np
from wandb.keras import WandbCallback

from flwr_p2p.keras.example import CreateMnistModel

from experiments.base_experimental_runner import BaseExperimentRunner


class NonFederatedRunner(BaseExperimentRunner):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def run(self):
        self.train_and_eval()

    def train_and_eval(self):
        image_size = self.x_train.shape[1]
        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        model_standalone1 = CreateMnistModel(self.config["lr"]).run()
        model_standalone2 = CreateMnistModel(self.config["lr"]).run()

        (
            partitioned_x_train,
            partitioned_y_train,
        ) = self.split_training_data_into_paritions(
            x_train, self.y_train, num_partitions=2
        )
        x_train_partition_1 = partitioned_x_train[0]
        y_train_partition_1 = partitioned_y_train[0]
        x_train_partition_2 = partitioned_x_train[1]
        y_train_partition_2 = partitioned_y_train[1]

        # Using generator for its ability to resume. This is important for federated
        # learning, otherwise in each federated round,
        # the cursor starts from the beginning every time.
        def train_generator1(batch_size):
            while True:
                for i in range(0, len(x_train_partition_1), batch_size):
                    yield x_train_partition_1[i : i + batch_size], y_train_partition_1[
                        i : i + batch_size
                    ]

        def train_generator2(batch_size):
            while True:
                for i in range(0, len(x_train_partition_2), batch_size):
                    yield x_train_partition_2[i : i + batch_size], y_train_partition_2[
                        i : i + batch_size
                    ]

        train_loader_standalone1 = train_generator1(self.config["batch_size"])
        train_loader_standalone2 = train_generator2(self.config["batch_size"])
        model_standalone1.fit(
            train_loader_standalone1,
            epochs=self.config["epochs"],
            steps_per_epoch=self.config["steps_per_epoch"],
            callbacks=[WandbCallback()],
        )
        model_standalone2.fit(
            train_loader_standalone2,
            epochs=self.config["epochs"],
            steps_per_epoch=self.config["steps_per_epoch"],
            callbacks=[WandbCallback()],
        )

        loss1, accuracy1 = model_standalone1.evaluate(
            x_test,
            self.y_test,
            batch_size=self.config["batch_size"],
            steps=self.config["steps_per_epoch"],
        )
        loss2, accuracy2 = model_standalone2.evaluate(
            x_test,
            self.y_test,
            batch_size=self.config["batch_size"],
            steps=self.config["steps_per_epoch"],
        )
