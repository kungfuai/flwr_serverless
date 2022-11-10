import numpy as np
import tensorflow as tf
import pytest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow import keras

from typing import List, Tuple
from numpy import array, float32

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
from flwr.client.numpy_client import NumPyClient
from uuid import uuid4

def split_training_data_into_paritions(x_train, y_train):
    # partion 1: classes 0-4
    # partion 2: classes 5-9
    # client 1 train on classes 0-4 only, and validated on 0-9
    # client 2 train on classes 5-9 only, and validated on 0-9
    # both clients will have low accuracy on 0-9 (below 0.6)
    # but when federated, the accuracy will be higher than 0.6
    partition_1_idx = y_train < 5
    partition_2_idx = y_train >= 5
    x_train_partition_1 = x_train[partition_1_idx]
    y_train_partition_1 = y_train[partition_1_idx]
    x_train_partition_2 = x_train[partition_2_idx]
    y_train_partition_2 = y_train[partition_2_idx]
    return x_train_partition_1, y_train_partition_1, x_train_partition_2, y_train_partition_2


def test_mnist_training_clients_on_partitioned_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 10
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    x_train_partition_1, y_train_partition_1, x_train_partition_2, y_train_partition_2 = split_training_data_into_paritions(x_train, y_train)

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
    model_standalone1.fit(train_loader_standalone1, epochs=3, steps_per_epoch=steps_per_epoch)
    model_standalone2.fit(train_loader_standalone2, epochs=3, steps_per_epoch=steps_per_epoch)
    _, accuracy_standalone1 = model_standalone1.evaluate(x_train, y_train, batch_size=batch_size, steps=10)
    _, accuracy_standalone2 = model_standalone2.evaluate(x_train, y_train, batch_size=batch_size, steps=10)
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

    num_federated_rounds = 3
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

    
    _, accuracy_federated = model_client1.evaluate(x_train, y_train, batch_size=32, steps=10)
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
    loss, accuracy = model.evaluate(x_train, y_train, batch_size=32, steps=10)
    # print(history[-1])
    assert accuracy > 0.6


def test_mnist_training_using_federated_nodes():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train.shape: (60000, 28, 28)
    # print(y_train.shape) # (60000,)
    # Normalize
    image_size = x_train.shape[1]
    batch_size = 32
    steps_per_epoch = 10
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    x_train_partition_1, y_train_partition_1, x_train_partition_2, y_train_partition_2 = split_training_data_into_paritions(x_train, y_train)

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
    model_standalone1.fit(train_loader_standalone1, epochs=3, steps_per_epoch=steps_per_epoch)
    model_standalone2.fit(train_loader_standalone2, epochs=3, steps_per_epoch=steps_per_epoch)
    _, accuracy_standalone1 = model_standalone1.evaluate(x_train, y_train, batch_size=batch_size, steps=10)
    _, accuracy_standalone2 = model_standalone2.evaluate(x_train, y_train, batch_size=batch_size, steps=10)
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

    num_federated_rounds = 3
    num_epochs_per_round = 1
    train_loader_client1 = train_generator1(batch_size=batch_size)
    train_loader_client2 = train_generator2(batch_size=batch_size)

    model_store = {
        # "client1/epoch1": parameters_1,
        # "client2/epoch1": parameters_2,
        # "latest_federated": parameters_latest,
    }
    """
    Synchronous version:

    8 am:
    client 1 (faster client) sends params1_1
    server has no params yet, so client 1 is told to wait
    server keeps params1_1

    9 am:
    client 2 (slower) sends params2_1 (client 1 is waiting from 8 am to 9 am)
    server aggregated params1_1 and params2_1, and sends back to client 1 and 2
    both client 1 and client 2 updates their local models, and resume training

    10 am:
    client 1: sends params1_2
    ...

    Asynchronous version (client does not wait for the server to get new aggregated weights):

    8 am:
    client 1 sends params1_1
    server returns params1_1, and sets params_federated_0 = params1_1
    client 1 keeps training with params1_1 for 2 hours

    9 am:
    client 2 sends params2_1
    server aggregates params1_1 and params2_1 into params_federated_1
    server returns aggregated params_federated_1
    client 2 updates its params to params_federated_1 and keeps training
    (but client 1 is busy doing its own training now, so it is not updated)

    10 am:
    client 1 sends params1_2
    server aggregates params_federated_1 and params1_2 into params_federated_2
    server returns aggregated params_federated_2
    client 1 updates its params to params_federated_2 and keeps training

    """

    class Node:
        def _should_wait(self):
            return False
        
        def update_parameters(self, current_parameters):
            pass

    class SynchronousNode:
        def __init__(self, num_clients=2):
            self.model_store = {}
            self.num_clients = num_clients

        def _should_wait(self):
            return len(self.model_store) == self.num_clients
        
        def update_parameters(self, current_parameters):
            pass

    class AsynchronousNode:
        def __init__(self, storage_backend = None):
            self.node_id = uuid4()
            self.counter = 0
            self.model_store = {
                "last_seen_node_id": None,
                "latest_federated": None,
            }
        
        def _get_latest_federated_model(self) -> Parameters:
            return self.model_store.get("latest_federated", None)
        
        def _aggregate(self, local_parameters: Parameters, federated_parameters: Parameters, num_examples: int = None, federated_num_examples: int = None) -> Parameters:
            if num_examples is None or federated_num_examples is None:
                num_examples = 1
                federated_num_examples = 1

            # Aggregation using the strategy.
            results: List[Tuple[ClientProxy, FitRes]] = [
                (
                    client_0,
                    FitRes(
                        status=Status(code=Code.OK, message="Success"),
                        parameters=local_parameters,
                        num_examples=num_examples,
                        metrics={},
                    ),
                ),
                (
                    client_1,
                    FitRes(
                        status=Status(code=Code.OK, message="Success"),
                        parameters=federated_parameters,
                        num_examples=federated_num_examples,
                        metrics={},
                    ),
                ),
            ]
            
            aggregated_parameters, _ = strategy.aggregate_fit(
                server_round=self.counter+1, results=results, failures=[]
            )
            self.counter += 1
            return aggregated_parameters

        def update_parameters(self, local_parameters, num_examples: int = None):
            latest_federated_parameters = self._get_latest_federated_model()
            if latest_federated_parameters is None:
                return None
            else:
                aggregated_parameters = self._aggregate(local_parameters, latest_federated_parameters)
                latest_federated_parameters = aggregated_parameters
                self.model_store["latest_federated"] = latest_federated_parameters
                return latest_federated_parameters


    node1 = AsynchronousNode()
    node2 = AsynchronousNode()
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        # TODO: bug! dataloader starts from the beginning of the dataset! We should use a generator
        model_client1.fit(train_loader_client1, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
        num_examples = batch_size * 10
        param_1: Parameters = ndarrays_to_parameters(model_client1.get_weights())
        updated_param_1 = node1.update_parameters(param_1, num_examples=num_examples)
        if updated_param_1 is not None:
            model_client1.set_weights(parameters_to_ndarrays(updated_param_1))

        model_client2.fit(train_loader_client2, epochs=num_epochs_per_round, steps_per_epoch=steps_per_epoch)
        num_examples = batch_size * 10
        param_2: Parameters = ndarrays_to_parameters(model_client2.get_weights())
        updated_param_2 = node2.update_parameters(param_2, num_examples=num_examples)
        if updated_param_2 is not None:
            model_client2.set_weights(parameters_to_ndarrays(updated_param_2))

    
    _, accuracy_federated = model_client1.evaluate(x_train, y_train, batch_size=32, steps=10)
    assert accuracy_federated > accuracy_standalone1
    assert accuracy_federated > accuracy_standalone2
    assert accuracy_federated > 0.6 # flaky test


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