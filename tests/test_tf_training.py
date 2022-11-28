from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import os
import time
from typing import List, Tuple, Any
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow import keras

from flwr.server.strategy import Strategy
from flwr.common import (
    Code,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM
from uuid import uuid4
from src.federated_node.async_federated_node import AsyncFederatedNode
from src.federated_node.sync_federated_node import SyncFederatedNode
from src.storage_backend.in_memory_storage_backend import InMemoryStorageBackend
from src.storage_backend.local_storage_backend import LocalStorageBackend
from src.keras.federated_learning_callback import FlwrFederatedCallback

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class FederatedLearningTestRun:
    num_nodes: int = 2
    epochs: int = 8
    num_rounds: int = 8  # number of federated rounds
    batch_size: int = 32
    steps_per_epoch: int = 10
    lr: float = 0.001
    test_steps: int = 10

    strategy: Strategy = FedAvg()
    storage_backend: Any = InMemoryStorageBackend()
    use_async_node: bool = True
    # Whether to train federated models concurrently or sequentially.
    train_concurrently: bool = False
    train_pseudo_concurrently: bool = False
    lag: float = 0.1

    def run(self):
        (
            self.partitioned_x_train,
            self.partitioned_y_train,
            self.x_test,
            self.y_test,
        ) = self.create_partitioned_datasets()
        model_standalone: List[keras.Model] = self.create_standalone_models()
        model_federated: List[keras.Model] = self.create_federated_models()
        model_standalone = self.train_standalone_models(model_standalone)
        model_federated = self.train_federated_models(model_federated)
        print("Evaluating on the combined test set (standalone models):")
        accuracy_standalone = self.evaluate_models(model_standalone)
        for i_node in range(len(accuracy_standalone)):
            print(
                "Standalone accuracy for node {}: {}".format(
                    i_node, accuracy_standalone[i_node]
                )
            )
        print("Evaluating on the combined test set (federated model):")
        # Evaluating only the first model.
        accuracy_federated = self.evaluate_models(model_federated)
        for i_node in range(self.num_nodes):  # [len(accuracy_federated) - 1]:
            print(
                "Federated accuracy for node {}: {}".format(
                    i_node, accuracy_federated[i_node]
                )
            )

        return accuracy_standalone, accuracy_federated

    def create_partitioned_datasets(self):
        num_partitions = self.num_nodes

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train.shape: (60000, 28, 28)
        # print(y_train.shape) # (60000,)
        # Normalize
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255
        partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
            x_train, y_train, num_partitions=num_partitions
        )
        return partitioned_x_train, partitioned_y_train, x_test, y_test

    def create_standalone_models(self):
        return [CreateMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]

    def get_train_dataloader_for_node(self, node_idx: int):
        partition_idx = node_idx
        batch_size = self.batch_size
        partitioned_x_train = self.partitioned_x_train
        partitioned_y_train = self.partitioned_y_train
        while True:
            for i in range(0, len(partitioned_x_train[partition_idx]), batch_size):
                yield partitioned_x_train[partition_idx][
                    i : i + batch_size
                ], partitioned_y_train[partition_idx][i : i + batch_size]

    def create_federated_models(self):
        return [CreateMnistModel(lr=self.lr).run() for _ in range(self.num_nodes)]

    def train_standalone_models(
        self, model_standalone: List[keras.Model]
    ) -> List[keras.Model]:
        for i_node in range(self.num_nodes):
            train_loader_standalone = self.get_train_dataloader_for_node(i_node)
            model_standalone[i_node].fit(
                train_loader_standalone,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
            )

        return model_standalone

    def train_federated_models(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        if self.train_pseudo_concurrently:
            print("Training federated models pseudo-concurrently.")
            return self._train_federated_models_pseudo_concurrently(model_federated)
        elif self.train_concurrently:
            print("Training federated models concurrently")
            return self._train_federated_models_concurrently(model_federated)
        else:
            print("Training federated models sequentially")
            return self._train_federated_models_sequentially(model_federated)

    def _train_federated_models_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = [
                AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy)
                for _ in range(self.num_nodes)
            ]
        else:
            nodes = [
                SyncFederatedNode(
                    storage_backend=storage_backend,
                    strategy=strategy,
                    num_nodes=self.num_nodes,
                )
                for _ in range(self.num_nodes)
            ]
        num_partitions = self.num_nodes
        model_federated = [
            CreateMnistModel(lr=self.lr).run() for _ in range(num_partitions)
        ]
        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i], epochs=self.epochs, x_test=self.x_test, y_test=self.y_test
            )
            for i in range(num_partitions)
        ]

        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        with ThreadPoolExecutor(max_workers=self.num_nodes) as ex:
            futures = []
            for i_node in range(self.num_nodes):
                # time.sleep(0.5 * i_node)
                future = ex.submit(
                    model_federated[i_node].fit,
                    x=train_loaders[i_node],
                    epochs=self.num_rounds,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=[callbacks_per_client[i_node]],
                    validation_data=(self.x_test, self.y_test),
                    validation_steps=self.test_steps,
                    validation_batch_size=self.batch_size,
                )
                futures.append(future)
            train_results = [future.result() for future in futures]

        return model_federated

    def _train_federated_models_pseudo_concurrently(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        # federated learning
        lag = self.lag
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = [
                AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy)
                for _ in range(self.num_nodes)
            ]
        else:
            raise NotImplementedError()
        num_partitions = self.num_nodes
        model_federated = [
            CreateMnistModel(lr=self.lr).run() for _ in range(num_partitions)
        ]
        callbacks_per_client = [
            FlwrFederatedCallback(
                nodes[i],
                num_examples_per_epoch=1000,
                x_test=self.x_test[: self.test_steps * self.batch_size, ...],
                y_test=self.y_test[: self.test_steps * self.batch_size, ...],
            )
            for i in range(num_partitions)
        ]

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        seqs = [[]] * self.num_nodes
        for i_node in range(self.num_nodes):
            seqs[i_node] = [
                (i_node, j + i_node * lag) for j in range(num_federated_rounds)
            ]
        # mix them up
        execution_sequence = []
        for i_node in range(self.num_nodes):
            execution_sequence.extend(seqs[i_node])
        execution_sequence = [
            x[0] for x in sorted(execution_sequence, key=lambda x: x[1])
        ]
        print(f"Execution sequence: {execution_sequence}")
        for i_node in execution_sequence:
            print("Training node", i_node)
            model_federated[i_node].fit(
                x=train_loaders[i_node],
                epochs=num_epochs_per_round,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=callbacks_per_client[i_node],
                validation_data=(
                    self.x_test[: self.test_steps * self.batch_size, ...],
                    self.y_test[: self.test_steps * self.batch_size, ...],
                ),
                validation_steps=self.test_steps,
                validation_batch_size=self.batch_size,
            )

            if i_node == 0:
                print("Evaluating on the combined test set:")
                model_federated[0].evaluate(
                    self.x_test[: self.test_steps * self.batch_size, ...],
                    self.y_test[: self.test_steps * self.batch_size, ...],
                    batch_size=self.batch_size,
                    steps=10,
                )

        return model_federated

    def _train_federated_models_sequentially(
        self, model_federated: List[keras.Model]
    ) -> List[keras.Model]:
        # federated learning
        strategy = self.strategy
        storage_backend = self.storage_backend
        if self.use_async_node:
            nodes = [
                AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy)
                for _ in range(self.num_nodes)
            ]
        else:
            raise NotImplementedError()
        num_partitions = self.num_nodes
        model_federated = [
            CreateMnistModel(lr=self.lr).run() for _ in range(num_partitions)
        ]
        callbacks_per_client = [
            FlwrFederatedCallback(nodes[i]) for i in range(num_partitions)
        ]

        num_federated_rounds = self.num_rounds
        num_epochs_per_round = 1
        train_loaders = [
            self.get_train_dataloader_for_node(i) for i in range(num_partitions)
        ]

        for i_round in range(num_federated_rounds):
            print("\n============ Round", i_round)
            for i_partition in range(num_partitions):
                model_federated[i_partition].fit(
                    train_loaders[i_partition],
                    epochs=num_epochs_per_round,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks_per_client[i_partition],
                )
            print("Evaluating on the combined test set:")
            model_federated[0].evaluate(
                self.x_test, self.y_test, batch_size=self.batch_size, steps=10
            )

        return model_federated

    def evaluate_models(self, models: List[keras.Model]) -> List[float]:
        accuracies = []
        for model in models:
            _, accuracy = model.evaluate(
                self.x_test,
                self.y_test,
                batch_size=self.batch_size,
                steps=self.test_steps,
            )
            accuracies.append(accuracy)
        return accuracies


class CreateMnistModel:
    def __init__(self, lr=0.001):
        self.lr = lr

    def run(self):
        model = self._build_model()
        return self._compile_model(model)

    def _build_model(self):
        input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=4, activation="relu")(input)
        x = MaxPooling2D()(x)
        x = Conv2D(16, kernel_size=4, activation="relu")(x)
        x = Flatten()(x)
        output = Dense(10, activation="softmax")(x)
        model = Model(inputs=input, outputs=output)
        return model

    def _compile_model(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def split_training_data_into_paritions(x_train, y_train, num_partitions: int = 2):
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


def test_mnist_training_clients_on_partitioned_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    epochs = 6
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 8
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
        x_train, y_train, num_partitions=2
    )
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
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

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(
        train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    model_standalone2.fit(
        train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    _, accuracy_standalone1 = model_standalone1.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    _, accuracy_standalone2 = model_standalone2.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    assert accuracy_standalone1 < 0.55
    assert accuracy_standalone2 < 0.55

    # federated learning
    model_client1 = CreateMnistModel().run()
    model_client2 = CreateMnistModel().run()

    # strategy = FedAvg()
    strategy = FedAvgM()
    # FedAdam does not work well in this setting.
    # tmp_model = CreateMnistModel().run()
    # strategy = FedAdam(initial_parameters=ndarrays_to_parameters(tmp_model.get_weights()), eta=1e-1)
    client_0 = None
    client_1 = None

    num_federated_rounds = epochs
    num_epochs_per_round = 1
    train_loader_client1 = train_generator1(batch_size=batch_size)
    train_loader_client2 = train_generator2(batch_size=batch_size)
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        # TODO: bug! dataloader starts from the beginning of the dataset! We should use a generator
        model_client1.fit(
            train_loader_client1,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        model_client2.fit(
            train_loader_client2,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10

        param_0: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        param_1: Parameters = ndarrays_to_parameters(model_client2.get_weights())

        # Aggregation using the strategy.
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                client_0,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_0,
                    num_examples=num_examples,
                    metrics={},
                ),
            ),
            (
                client_1,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_1,
                    num_examples=num_examples,
                    metrics={},
                ),
            ),
        ]

        aggregated_parameters, _ = strategy.aggregate_fit(
            server_round=i_round + 1, results=results, failures=[]
        )
        # turn actual_aggregated back to keras.Model.
        aggregated_parameters_numpy: NDArrays = parameters_to_ndarrays(
            aggregated_parameters
        )
        # Update client model weights using the aggregated parameters.
        model_client1.set_weights(aggregated_parameters_numpy)
        model_client2.set_weights(aggregated_parameters_numpy)

    _, accuracy_federated = model_client1.evaluate(
        x_test, y_test, batch_size=32, steps=10
    )
    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6  # flaky test


def test_mnist_training_standalone():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    model = CreateMnistModel().run()

    model.fit(x_train, y_train, epochs=3, batch_size=32, steps_per_epoch=10)
    # TODO: look into the history object to get accuracy
    # memorization test
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32, steps=10)
    # print(history[-1])
    assert accuracy > 0.6


def test_mnist_training_using_federated_nodes():
    # epochs = standalone_epochs = 3  # does not work
    epochs = standalone_epochs = 8  # works

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 8

    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(
        x_train, y_train, num_partitions=2
    )
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
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

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(
        train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    model_standalone2.fit(
        train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    print("Evaluating on the combined test set:")
    _, accuracy_standalone1 = model_standalone1.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    _, accuracy_standalone2 = model_standalone2.evaluate(
        x_test, y_test, batch_size=batch_size, steps=10
    )
    assert accuracy_standalone1 < 0.55
    assert accuracy_standalone2 < 0.55

    # federated learning
    model_client1 = CreateMnistModel().run()
    model_client2 = CreateMnistModel().run()

    strategy = FedAvg()
    # strategy = FedAvgM()
    # FedAdam does not work well in this setting.
    # tmp_model = CreateMnistModel().run()
    # strategy = FedAdam(initial_parameters=ndarrays_to_parameters(tmp_model.get_weights()), eta=1e-1)

    num_federated_rounds = standalone_epochs
    num_epochs_per_round = 1
    train_loader_client1 = train_generator1(batch_size=batch_size)
    train_loader_client2 = train_generator2(batch_size=batch_size)

    storage_backend = InMemoryStorageBackend()
    node1 = AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy)
    node2 = AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy)
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        model_client1.fit(
            train_loader_client1,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10
        param_1: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        updated_param_1 = node1.update_parameters(param_1, num_examples=num_examples)
        if updated_param_1 is not None:
            model_client1.set_weights(parameters_to_ndarrays(updated_param_1))
        else:
            print("node1 is waiting for other nodes to send their parameters")

        model_client2.fit(
            train_loader_client2,
            epochs=num_epochs_per_round,
            steps_per_epoch=steps_per_epoch,
        )
        num_examples = batch_size * 10
        param_2: Parameters = ndarrays_to_parameters(model_client2.get_weights())
        updated_param_2 = node2.update_parameters(param_2, num_examples=num_examples)
        if updated_param_2 is not None:
            model_client2.set_weights(parameters_to_ndarrays(updated_param_2))
        else:
            print("node2 is waiting for other nodes to send their parameters")

        print("Evaluating on the combined test set:")
        _, accuracy_federated = model_client1.evaluate(
            x_test, y_test, batch_size=32, steps=10
        )

    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6  # flaky test


def test_mnist_federated_callback_2nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05


def test_mnist_federated_callback_3nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=3,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05


def test_mnist_federated_callback_2nodes_lag0_1(tmpdir):
    epochs = 10
    num_nodes = 2
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        # storage_backend=InMemoryStorageBackend(),
        storage_backend=LocalStorageBackend(directory=str(tmpdir.join("fed_test"))),
        train_pseudo_concurrently=True,
        use_async_node=True,
        lag=0.1,
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


def test_mnist_federated_callback_2nodes_lag2(tmpdir):
    epochs = 10
    num_nodes = 2
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        storage_backend=InMemoryStorageBackend(),
        # storage_backend=LocalStorageBackend(directory=str(tmpdir.join("fed_test"))),
        train_pseudo_concurrently=True,
        use_async_node=True,
        lag=2,
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


def test_mnist_federated_callback_2nodes_concurrent(tmpdir):
    epochs = 8
    num_nodes = 2
    fed_dir = tmpdir.join("fed_test")
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=num_nodes,
        epochs=epochs,
        num_rounds=epochs,
        batch_size=32,
        steps_per_epoch=8,
        lr=0.001,
        strategy=FedAvg(),
        # storage_backend=InMemoryStorageBackend(),
        storage_backend=LocalStorageBackend(directory=str(fed_dir)),
        train_concurrently=True,
        # use_async_node=False,
        use_async_node=True,
    ).run()
    # print(fed_dir.listdir())
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[-1] > accuracy_standalone[-1]
    assert accuracy_federated[-1] > 1.0 / num_nodes + 0.05


if __name__ == "__main__":
    test_mnist_federated_callback_2nodes_concurrent()
