import os
from io import BytesIO
import json
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
        x_test=None,
        y_test=None,
        test_batch_size=32,
        test_steps=10,
        override_metrics_with_aggregated_metrics: bool = False,
        save_model_before_aggregation: bool = False,
        save_model_after_aggregation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node = node
        self.num_examples_per_epoch = num_examples_per_epoch
        self.override_metrics_with_aggregated_metrics = (
            override_metrics_with_aggregated_metrics
        )
        self.save_model_before_aggregation = save_model_before_aggregation
        self.save_model_after_aggregation = save_model_after_aggregation
        self.x_test = x_test
        self.y_test = y_test
        self.test_batch_size = test_batch_size
        self.test_steps = test_steps
        self.model_before_aggregation_filename_pattern = (
            "keras/{node_id}/model_before_aggregation_{epoch:05d}.h5"
        )
        self.metrics_before_aggregation_filename_pattern = (
            "keras/{node_id}/metrics_before_aggregation_{epoch:05d}.json"
        )
        self.model_after_aggregation_filename_pattern = (
            "keras/{node_id}/model_after_aggregation_{epoch:05d}.h5"
        )
        self.metrics_after_aggregation_filename_pattern = (
            "keras/{node_id}/metrics_after_aggregation_{epoch:05d}.json"
        )

    def _save_model_to_shared_folder(self, filename: str):
        folder = self.node.model_store.get_raw_folder()
        key = filename
        # convert model into bytes
        tmp_path = f"tmp_model_{self.node.node_id}.h5"
        self.model.save(tmp_path)
        with open(tmp_path, "rb") as f:
            model_bytes = f.read()
        folder[key] = model_bytes
        # delete
        os.remove(tmp_path)

    def _save_metrics_to_shared_folder(self, filename: str, metrics: dict):
        folder = self.node.model_store.get_raw_folder()
        key = filename
        metrics_bytes = BytesIO()
        json_str = json.dumps(metrics, indent=2)
        metrics_bytes.write(json_str.encode("utf-8"))
        folder[key] = metrics_bytes.getvalue()

    def on_epoch_end(self, epoch: int, logs=None):
        # use the P2PStrategy to update the model.
        node_id = self.node.node_id

        if logs:
            # Save metrics.
            filename = self.metrics_before_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_metrics_to_shared_folder(filename, logs)

        if self.save_model_before_aggregation:
            filename = self.model_before_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_model_to_shared_folder(filename)

        params: Parameters = ndarrays_to_parameters(self.model.get_weights())
        updated_params, updated_metrics = self.node.update_parameters(
            params, num_examples=self.num_examples_per_epoch, epoch=epoch
        )

        # save metrics after aggregation
        if updated_metrics:
            filename = self.metrics_after_aggregation_filename_pattern.format(
                node_id=node_id, epoch=epoch
            )
            self._save_metrics_to_shared_folder(filename, updated_metrics)

        if updated_params is not None:
            self.model.set_weights(parameters_to_ndarrays(updated_params))
            if self.save_model_before_aggregation:
                node_id = self.node.node_id
                filename = self.model_before_aggregation_filename_pattern.format(
                    node_id=node_id, epoch=epoch
                )
                self._save_model_to_shared_folder(filename)
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
