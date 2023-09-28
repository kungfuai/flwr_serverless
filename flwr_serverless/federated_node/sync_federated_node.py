from typing import List, Tuple
from uuid import uuid4
import time
import logging
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
)
from flwr.server.client_proxy import ClientProxy
from .aggregatable import Aggregatable


LOGGER = logging.getLogger(__name__)


class SyncFederatedNode:
    """
    Synchronous federated learning.

    TODO: allow user to specify the metric names to include and exclude.
    """

    def __init__(self, shared_folder, strategy, num_nodes: int):
        self.node_id = str(uuid4())
        self.counter = 0
        self.strategy = strategy
        self.model_store = shared_folder
        self.seen_models = set()
        self.num_nodes = num_nodes

    def _aggregate(self, aggregatables: List[Aggregatable]) -> Aggregatable:
        # Aggregation using the flwr strategy.
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=param_holder.parameters,
                    num_examples=param_holder.num_examples,
                    metrics=param_holder.metrics,
                ),
            )
            for param_holder in aggregatables
        ]

        aggregated_parameters, aggregated_metrics = self.strategy.aggregate_fit(
            server_round=self.counter + 1, results=results, failures=[]
        )
        aggregated_metrics = self._update_aggregated_metrics_in_case_flwr_did_not_do_it(
            aggregatables, aggregated_metrics
        )

        self.counter += 1
        return Aggregatable(
            parameters=aggregated_parameters,
            num_examples=sum(
                [param_holder.num_examples for param_holder in aggregatables]
            ),
            metrics=aggregated_metrics,
        )

    def _update_aggregated_metrics_in_case_flwr_did_not_do_it(
        self, aggregatables, aggregated_metrics: dict
    ) -> dict:
        if len(aggregated_metrics) == 0:
            aggregated_metrics = {}
            aggregated_metrics["num_examples"] = sum(
                [param_holder.num_examples for param_holder in aggregatables]
            )
            aggregated_metrics["num_nodes"] = len(aggregatables)
            first_metric = aggregatables[0].metrics
            for k, _ in first_metric.items():
                if k in ["num_nodes", "num_examples"]:
                    continue
                aggregated_metrics[k] = (
                    sum(
                        [
                            param_holder.metrics[k] * param_holder.num_examples
                            for param_holder in aggregatables
                        ]
                    )
                    / aggregated_metrics["num_examples"]
                )
        LOGGER.info(f"Aggregated metrics: {aggregated_metrics}")
        return aggregated_metrics

    def _get_parameters_from_other_nodes(self, epoch: int) -> List[Aggregatable]:
        other_parameters_from_epoch = []

        # with open("logs/model_store.txt", "a") as f:
        #     f.write(f"Current model_store for {self.node_id} on epoch {epoch}:\n")
        #     for key, value in self.model_store.items():
        #         f.write(
        #             f"key: {key}, epoch: {value['epoch']}, node_id: {self.node_id}\n"
        #         )

        for key, value in self.model_store.items():
            if not isinstance(value, dict):
                continue
            if "epoch" not in value:
                raise KeyError(f"epoch not in the dictionary: {value.keys()}")
            if value["epoch"] != epoch or value["node_id"] == self.node_id:
                continue
            other_parameters_from_epoch.append(value["aggregatable"])

        return other_parameters_from_epoch

    def update_parameters(
        self,
        local_parameters: Parameters,
        num_examples: int = None,
        metrics: dict = None,
        epoch: int = None,
        upload_only=False,
    ) -> Tuple[Parameters, dict]:
        model_hash = self.node_id + "_" + str(time.time())
        self_aggregatable = Aggregatable(
            parameters=local_parameters,
            num_examples=num_examples,
            metrics=metrics,
        )
        self.model_store[model_hash] = dict(
            aggregatable=self_aggregatable,
            model_hash=model_hash,
            epoch=epoch,
            node_id=self.node_id,
        )
        # if len(self.model_store) > self.num_nodes:
        #     raise ValueError(
        #         f"Too many nodes in the federated learning run: {len(self.model_store)}. Expected {self.num_nodes}"
        #     )
        if upload_only:
            return None
        aggregatables_from_other_nodes = self._get_parameters_from_other_nodes(epoch)
        wait_counter = 0
        max_retry = 3  # 60 * 10
        while len(aggregatables_from_other_nodes) < self.num_nodes - 1:
            # Other nodes have not all sent their parameters yet.
            # Wait with exponential back-off.
            LOGGER.info(
                f"Got {len(aggregatables_from_other_nodes)} parameters from other nodes."
            )
            LOGGER.info(
                f"Waiting for {self.num_nodes - 1 - len(aggregatables_from_other_nodes)} more."
            )
            wait_seconds = min(60 * 2**wait_counter, max_retry)
            LOGGER.info(f"Waiting {wait_seconds} seconds..")
            time.sleep(wait_seconds)
            wait_counter += 1
            aggregatables_from_other_nodes = self._get_parameters_from_other_nodes(
                epoch
            )

        # Aggregate the parameters from other nodes
        parameters_from_all_nodes = aggregatables_from_other_nodes + [self_aggregatable]
        aggregated_parameters_and_metrics = self._aggregate(parameters_from_all_nodes)
        # self.model_store["latest_federated"] = aggregated_parameters
        return (
            aggregated_parameters_and_metrics.parameters,
            aggregated_parameters_and_metrics.metrics,
        )
