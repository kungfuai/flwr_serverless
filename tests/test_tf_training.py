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
from flwr.server.grpc_server.grpc_client_proxy import GrpcClientProxy
from flwr.server.strategy import FedAvg


def split_training_data_into_paritions(x_train, y_train):
    # partion 1: 0-4
    # partion 2: 5-9
    # client 1 train on 0-4 only, and validated on 0-9
    # client 2 train on 5-9 only, and validated on 0-9
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
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    model_client1 = CreateMnistModel().run()
    model_client2 = CreateMnistModel().run()

    x_train_partition_1, y_train_partition_1, x_train_partition_2, y_train_partition_2 = split_training_data_into_paritions(x_train, y_train)

    model_client1.fit(x_train_partition_1, y_train_partition_1, epochs=3, batch_size=32, steps_per_epoch=10)
    model_client2.fit(x_train_partition_2, y_train_partition_2, epochs=3, batch_size=32, steps_per_epoch=10)
    _, accuracy_client1 = model_client1.evaluate(x_train, y_train, batch_size=32, steps=10)
    _, accuracy_client2 = model_client2.evaluate(x_train, y_train, batch_size=32, steps=10)
    assert accuracy_client1 < 0.6
    assert accuracy_client2 < 0.6

    # federated learning

    # param_0: Parameters = ndarrays_to_parameters(
    #     [model_client1.get_weights()]
    # )
    param_0: Parameters = fl.common.weights_to_parameters(model_client1.get_weights())
    param_1: Parameters = fl.common.weights_to_parameters(model_client2.get_weights())

    client_0 = None
    client_1 = None
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            client_0,
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_0,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            client_1,
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_1,
                num_examples=5,
                metrics={},
            ),
        ),
    ]

    strategy = FedAvg()
    actual_aggregated, _ = strategy.aggregate_fit(
        server_round=1, results=results, failures=[]
    )
    # TODO: turn it back to keras.Model.


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

    # class MnistClient(fl.client.NumPyClient):
    #     def get_parameters(self):
    #         return model.get_weights()
    #     def fit(self, parameters, config):
    #         model.set_weights(parameters)
    #         model.fit(x_train, y_train, epochs=10, batch_size=32, steps_per_epoch=3)
    #         return model.get_weights(), len(x_train), {}
    #     def evaluate(self, parameters, config):
    #         model.set_weights(parameters)
    #         loss, accuracy = model.evaluate(x_test, y_test)
    #         return loss, len(x_test), {“accuracy”: accuracy}
    # fl.client.start_numpy_client(“[::]:8080", client=MnistClient())



@pytest.mark.skip(reason="Not implemented yet")
def test_mnist_training_through_flwr_strategy_with_partitioned_classes():
    pass


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