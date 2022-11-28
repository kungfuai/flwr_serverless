from tensorflow import keras
from src.federated_node.async_federated_node import AsyncFederatedNode
from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FlwrFederatedCallback(keras.callbacks.Callback):
    def __init__(
        self,
        node: AsyncFederatedNode,
        epochs: int = None,
        num_examples_per_epoch: int = None,
        x_test=None,
        y_test=None,
        **kwargs,
    ):
        """
        TODO: User needs to specify a shared folder / bucket.
        User optionally can specify a strategy by name.
        """
        super().__init__(**kwargs)
        self.node = node
        self.epochs = epochs
        self.num_examples_per_epoch = num_examples_per_epoch
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch: int, logs=None):
        # use the P2PStrategy to update the model.
        param_1: Parameters = ndarrays_to_parameters(self.model.get_weights())
        if self.epochs is not None and epoch == self.epochs - 1:
            upload_only = True
        else:
            upload_only = False
        updated_param_1 = self.node.update_parameters(
            param_1, upload_only=upload_only, num_examples=self.num_examples_per_epoch
        )
        if updated_param_1 is not None:
            self.model.set_weights(parameters_to_ndarrays(updated_param_1))
            if self.x_test is not None:
                print("\n=========================== eval inside callback")
                self.model.evaluate(
                    self.x_test, self.y_test, batch_size=32, steps=10, verbose=2
                )
        else:
            print("waiting for other nodes to send their parameters")
