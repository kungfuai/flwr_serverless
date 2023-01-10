import numpy as np
from wandb.keras import WandbCallback

from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr_p2p.federated_node.async_federated_node import AsyncFederatedNode
from flwr_p2p.shared_folder.in_memory_folder import InMemoryFolder
from flwr_p2p.keras.example import CreateMnistModel

from experiments.base_experimental_model import BaseExperimentRunner


class FederatedLearningRunner(BaseExperimentRunner):
    def __init__(self, config, dataset, strategy):
        super().__init__(config, dataset)
        if strategy == "fedavg":
            self.strategy = FedAvg()
        elif strategy == "fedavgm":
            self.strategy = FedAvgM()
        elif strategy == "fedadam":
            self.strategy = FedAdam()
        else:
            raise ValueError("Strategy not supported")

    def run(self):
        self.fed_async_train_and_eval()

    def fed_async_train_and_eval(self):
        image_size = self.x_train.shape[1]
        # x_train.shape: (60000, 28, 28)
        # print(y_train.shape) # (60000,)
        # Normalize

        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

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

        # federated learning
        model_client1 = CreateMnistModel(lr=0.0004).run()
        model_client2 = CreateMnistModel(lr=0.0004).run()

        num_federated_rounds = self.config["epochs"]
        num_epochs_per_round = 1
        train_loader_client1 = train_generator1(batch_size=self.config["batch_size"])
        train_loader_client2 = train_generator2(batch_size=self.config["batch_size"])

        storage_backend = InMemoryFolder()
        node1 = AsyncFederatedNode(
            shared_folder=storage_backend, strategy=self.strategy
        )
        node2 = AsyncFederatedNode(
            shared_folder=storage_backend, strategy=self.strategy
        )
        for i_round in range(num_federated_rounds):
            print("\n============ Round", i_round)
            model_client1.fit(
                train_loader_client1,
                epochs=num_epochs_per_round,
                steps_per_epoch=self.config["steps_per_epoch"],
                callbacks=[WandbCallback()],
            )
            num_examples = self.config["batch_size"] * self.config["steps_per_epoch"]
            param_1: Parameters = ndarrays_to_parameters(model_client1.get_weights())
            updated_param_1 = node1.update_parameters(
                param_1, num_examples=num_examples
            )
            if updated_param_1 is not None:
                model_client1.set_weights(parameters_to_ndarrays(updated_param_1))
            else:
                print("node1 is waiting for other nodes to send their parameters")

            model_client2.fit(
                train_loader_client2,
                epochs=num_epochs_per_round,
                steps_per_epoch=self.config["steps_per_epoch"],
                callbacks=[WandbCallback()],
            )
            num_examples = self.config["batch_size"] * self.config["steps_per_epoch"]
            param_2: Parameters = ndarrays_to_parameters(model_client2.get_weights())
            updated_param_2 = node2.update_parameters(
                param_2, num_examples=num_examples
            )
            if updated_param_2 is not None:
                model_client2.set_weights(parameters_to_ndarrays(updated_param_2))
            else:
                print("node2 is waiting for other nodes to send their parameters")

            print("Evaluating on the combined test set:")
            _, accuracy_federated = model_client1.evaluate(
                x_test, self.y_test, batch_size=32, steps=self.config["steps_per_epoch"]
            )

        print("accuracy_federated", accuracy_federated)
