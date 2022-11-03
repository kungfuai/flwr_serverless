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
from flwr.server.strategy import FedAvg, FedAdam
from flwr.client.numpy_client import NumPyClient


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

    model_standalone1 = CreateMnistModel().run()
    model_standalone2 = CreateMnistModel().run()

    x_train_partition_1, y_train_partition_1, x_train_partition_2, y_train_partition_2 = split_training_data_into_paritions(x_train, y_train)

    model_standalone1.fit(x_train_partition_1, y_train_partition_1, epochs=3, batch_size=32, steps_per_epoch=10)
    model_standalone2.fit(x_train_partition_2, y_train_partition_2, epochs=3, batch_size=32, steps_per_epoch=10)
    _, accuracy_standalone1 = model_standalone1.evaluate(x_train, y_train, batch_size=32, steps=10)
    _, accuracy_standalone2 = model_standalone2.evaluate(x_train, y_train, batch_size=32, steps=10)
    assert accuracy_standalone1 < 0.55
    assert accuracy_standalone2 < 0.55

    # federated learning
    model_client1 = CreateMnistModel().run()
    model_client2 = CreateMnistModel().run()

    # strategy = FedAvg()
    tmp_model = CreateMnistModel().run()
    strategy = FedAdam(initial_parameters=ndarrays_to_parameters(tmp_model.get_weights()), eta=0.01, eta_l=0.001)
    client_0 = None
    client_1 = None

    num_federated_rounds = 5
    num_epochs_per_round = 1
    for i_round in range(num_federated_rounds):
        print("\n============ Round", i_round)
        # TODO: bug! dataloader starts from the beginning of the dataset! We should use a generator
        model_client1.fit(x_train_partition_1, y_train_partition_1, epochs=num_epochs_per_round, batch_size=32, steps_per_epoch=10)
        model_client2.fit(x_train_partition_2, y_train_partition_2, epochs=num_epochs_per_round, batch_size=32, steps_per_epoch=10)
        num_examples = 32 * 10

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
    # assert accuracy_federated > 0.55 # flaky test


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