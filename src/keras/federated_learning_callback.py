from tensorflow import keras
from src.federated_node.async_federated_node import AsyncFederatedNode
from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FlwrFederatedCallback(keras.callbacks.Callback):
    def __init__(self, node: AsyncFederatedNode, **kwargs):
        """
        TODO: User needs to specify a shared folder / bucket.
        User optionally can specify a strategy by name.
        """
        super().__init__(**kwargs)
        self.node = node

    def on_epoch_end(self, epoch, logs=None):
        # use the P2PStrategy to update the model.
        param_1: Parameters = ndarrays_to_parameters(self.model.get_weights())
        updated_param_1 = self.node.update_parameters(param_1)
        if updated_param_1 is not None:
            self.model.set_weights(parameters_to_ndarrays(updated_param_1))
        else:
            print("node1 is waiting for other nodes to send their parameters")
