from tensorflow import keras
from flwr_serverless.federated_node.async_federated_node import AsyncFederatedNode
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
        num_examples_per_epoch: int,
        override_metrics_with_aggregated_metrics: bool = False,
        x_test=None,
        y_test=None,
        test_batch_size=32,
        test_steps=10,
        **kwargs,
    ):
        """
        TODO: User needs to specify a shared folder / bucket.
        User optionally can specify a strategy by name.
        """
        super().__init__(**kwargs)
        self.node = node
        self.num_examples_per_epoch = num_examples_per_epoch
        self.override_metrics_with_aggregated_metrics = (
            override_metrics_with_aggregated_metrics
        )
        self.x_test = x_test
        self.y_test = y_test
        self.test_batch_size = test_batch_size
        self.test_steps = test_steps

    def on_epoch_end(self, epoch: int, logs=None):
        # use the P2PStrategy to update the model.
        params: Parameters = ndarrays_to_parameters(self.model.get_weights())
        updated_params, updated_metrics = self.node.update_parameters(
            params, num_examples=self.num_examples_per_epoch, epoch=epoch
        )
        if updated_params is not None:
            self.model.set_weights(parameters_to_ndarrays(updated_params))
            if self.override_metrics_with_aggregated_metrics:
                if updated_metrics is not None:
                    logs.update(updated_metrics)
            if self.x_test is not None:
                print("\n=========================== eval inside callback")
                self.model.evaluate(
                    self.x_test,
                    self.y_test,
                    batch_size=self.test_batch_size,
                    steps=self.test_steps,
                    verbose=2,
                )
                print("Done evaluating inside callback =====================\n")
        else:
            print("waiting for other nodes to send their parameters")