import numpy as np
import tensorflow as tf
import pytest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow import keras

from typing import List, Tuple

import flwr as fl
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
from src.storage_backend.in_memory_storage_backend import InMemoryStorageBackend
from src.keras.federated_learning_callback import FlwrFederatedCallback


def split_training_data_into_paritions(x_train, y_train, num_partitions: int = 2):
    # partion 1: classes 0-4
    # partion 2: classes 5-9
    # client 1 train on classes 0-4 only, and validated on 0-9
    # client 2 train on classes 5-9 only, and validated on 0-9
    # both clients will have low accuracy on 0-9 (below 0.6)
    # but when federated, the accuracy will be higher than 0.6
    classes = list(range(10))
    num_classes_per_partition = int(len(classes) / num_partitions)
    partitioned_classes = [classes[i:i + num_classes_per_partition] for i in range(0, len(classes), num_classes_per_partition)]
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
    epochs = 5
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 10
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(x_train, y_train, num_partitions=2)
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
    # the cursor starts from the beginning every time.
    def train_generator1(batch_size):
        while True:
            for i in range(0, len(x_train_partition_1), batch_size):
                yield x_train_partition_1[i:i+batch_size], y_train_partition_1[i:i+batch_size]
    
    def train_generator2(batch_size):
        while True:
            for i in range(0, len(x_train_partition_2), batch_size):
                yield x_train_partition_2[i:i+batch_size], y_train_partition_2[i:i+batch_size]

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    model_standalone2.fit(train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch)
    _, accuracy_standalone1 = model_standalone1.evaluate(x_test, y_test, batch_size=batch_size, steps=10)
    _, accuracy_standalone2 = model_standalone2.evaluate(x_test, y_test, batch_size=batch_size, steps=10)
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
        model_client1.fit(train_loader_client1, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
        model_client2.fit(train_loader_client2, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
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
            server_round=i_round+1, results=results, failures=[]
        )
        # turn actual_aggregated back to keras.Model.
        aggregated_parameters_numpy: NDArrays = parameters_to_ndarrays(aggregated_parameters)
        # Update client model weights using the aggregated parameters.
        model_client1.set_weights(aggregated_parameters_numpy)
        model_client2.set_weights(aggregated_parameters_numpy)

    
    _, accuracy_federated = model_client1.evaluate(x_test, y_test, batch_size=32, steps=10)
    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6 # flaky test


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
    epochs = standalone_epochs = 6  # works

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

    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(x_train, y_train, num_partitions=2)
    x_train_partition_1 = partitioned_x_train[0]
    y_train_partition_1 = partitioned_y_train[0]
    x_train_partition_2 = partitioned_x_train[1]
    y_train_partition_2 = partitioned_y_train[1]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
    # the cursor starts from the beginning every time.
    def train_generator1(batch_size):
        while True:
            for i in range(0, len(x_train_partition_1), batch_size):
                yield x_train_partition_1[i:i+batch_size], y_train_partition_1[i:i+batch_size]
    
    def train_generator2(batch_size):
        while True:
            for i in range(0, len(x_train_partition_2), batch_size):
                yield x_train_partition_2[i:i+batch_size], y_train_partition_2[i:i+batch_size]

    train_loader_standalone1 = train_generator1(batch_size)
    train_loader_standalone2 = train_generator2(batch_size)
    model_standalone1.fit(train_loader_standalone1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    model_standalone2.fit(train_loader_standalone2, epochs=epochs, steps_per_epoch=steps_per_epoch)
    print("Evaluating on the combined test set:")
    _, accuracy_standalone1 = model_standalone1.evaluate(x_test, y_test, batch_size=batch_size, steps=10)
    _, accuracy_standalone2 = model_standalone2.evaluate(x_test, y_test, batch_size=batch_size, steps=10)
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
        model_client1.fit(train_loader_client1, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
        num_examples = batch_size * 10
        param_1: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        updated_param_1 = node1.update_parameters(param_1, num_examples=num_examples)
        if updated_param_1 is not None:
            model_client1.set_weights(parameters_to_ndarrays(updated_param_1))
        else:
            print("node1 is waiting for other nodes to send their parameters")

        model_client2.fit(train_loader_client2, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
        num_examples = batch_size * 10
        param_2: Parameters = ndarrays_to_parameters(model_client2.get_weights())
        updated_param_2 = node2.update_parameters(param_2, num_examples=num_examples)
        if updated_param_2 is not None:
            model_client2.set_weights(parameters_to_ndarrays(updated_param_2))
        else:
            print("node2 is waiting for other nodes to send their parameters")

    print("Evaluating on the combined test set:")
    _, accuracy_federated = model_client1.evaluate(x_test, y_test, batch_size=32, steps=10)
    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6 # flaky test


def test_mnist_federated_callback():
    num_partitions = 3
    # epochs = standalone_epochs = 3  # does not work
    epochs = standalone_epochs = 10  # works

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
    partitioned_x_train, partitioned_y_train = split_training_data_into_paritions(x_train, y_train, num_partitions=num_partitions)

    model_standalone = [CreateMnistModel().run() for _ in range(num_partitions)]

    # Using generator for its ability to resume. This is important for federated learning, otherwise in each federated round,
    # the cursor starts from the beginning every time.
    def train_generator(partition_idx: int):
        while True:
            for i in range(0, len(partitioned_x_train[partition_idx]), batch_size):
                yield partitioned_x_train[partition_idx][i:i+batch_size], partitioned_y_train[partition_idx][i:i+batch_size]

    for i_partition in range(num_partitions):
        train_loader_standalone = train_generator(i_partition)
        model_standalone[i_partition].fit(train_loader_standalone, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    print("Evaluating on the combined test set:")
    accuracy_standalone = [0] * num_partitions
    for i_partition in range(num_partitions):
        _, accuracy_standalone[i_partition] = model_standalone[i_partition].evaluate(x_test, y_test, batch_size=batch_size, steps=10)
        assert accuracy_standalone[i_partition] < 1. / num_partitions + 0.05

    # federated learning
    strategy = FedAvg()
    storage_backend = InMemoryStorageBackend()
    nodes = [AsyncFederatedNode(storage_backend=storage_backend, strategy=strategy) for _ in range(num_partitions)]
    model_federated = [CreateMnistModel().run() for _ in range(num_partitions)]
    callbacks_per_client = [FlwrFederatedCallback(nodes[i]) for i in range(num_partitions)]

    num_federated_rounds = standalone_epochs
    num_epochs_per_round = 1
    train_loaders = [train_generator(i) for i in range(num_partitions)]

    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        for i_partition in range(num_partitions):
            model_federated[i_partition].fit(
                train_loaders[i_partition],
                epochs=num_epochs_per_round,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks_per_client[i_partition]
            )
        print("Evaluating on the combined test set:")
        _, accuracy_federated = model_federated[0].evaluate(x_test, y_test, batch_size=batch_size, steps=10)
        
    assert accuracy_federated > accuracy_standalone[0]
    assert accuracy_federated > 0.5


class CreateMnistModel:
    def __init__(self):
        pass

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
            optimizer=keras.optimizers.Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model