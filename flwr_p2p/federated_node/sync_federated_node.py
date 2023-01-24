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


LOGGER = logging.getLogger(__name__)


class SyncFederatedNode:
    """
    Synchronous federated learning.
    """

    def __init__(self, shared_folder, strategy, num_nodes: int):
        self.node_id = str(uuid4())
        self.counter = 0
        self.strategy = strategy
        self.model_store = shared_folder
        self.seen_models = set()
        self.num_nodes = num_nodes

    def _aggregate(
        self, parameters_list: List[Parameters], num_examples_list: List[int]
    ) -> Parameters:
        # Aggregation using the flwr strategy.
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=p,
                    num_examples=num_examples,
                    metrics={},
                ),
            )
            for p, num_examples in zip(parameters_list, num_examples_list)
        ]

        aggregated_parameters, _ = self.strategy.aggregate_fit(
            server_round=self.counter + 1, results=results, failures=[]
        )
        self.counter += 1
        return aggregated_parameters

    def _get_parameters_from_other_nodes(self, epoch) -> List[Parameters]:
        other_parameters_from_epoch = []

        # with open("logs/model_store.txt", "a") as f:
        #     f.write(f"Current model_store for {self.node_id} on epoch {epoch}:\n")
        #     for key, value in self.model_store.items():
        #         f.write(
        #             f"key: {key}, epoch: {value['epoch']}, node_id: {self.node_id}\n"
        #         )

        for key, value in self.model_store.items():
            if value["epoch"] != epoch or value["node_id"] == self.node_id:
                continue
            other_parameters_from_epoch.append(value["parameters"])

        return other_parameters_from_epoch
        # unseen_parameters_from_other_nodes = []
        # for key, value in self.model_store.items():
        #     print("\n------------------------==================")
        #     print(self.node_id, key)
        #     print("==================-----------------------")
        #     if isinstance(value, dict) and "parameters" in value:
        #         if key != self.node_id:
        #             model_hash = value["model_hash"]
        #             if model_hash not in self.seen_models:
        #                 self.seen_models.add(model_hash)
        #                 unseen_parameters_from_other_nodes.append(value["parameters"])
        # return unseen_parameters_from_other_nodes

    def update_parameters(
        self,
        local_parameters: Parameters,
        upload_only=False,
        num_examples: int = None,
        epoch: int = None,
    ):
        model_hash = self.node_id + str(time.time())
        self.model_store[model_hash] = dict(
            parameters=local_parameters,
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
        # print(f"\n{len(self.model_store)} nodes\n")
        parameters_from_other_nodes = self._get_parameters_from_other_nodes(epoch)
        wait_counter = 0
        max_wait = 3  # 60 * 10
        while len(parameters_from_other_nodes) < self.num_nodes - 1:
            # Other nodes have not all sent their parameters yet.
            # Wait with exponential back-off.
            LOGGER.info(
                f"Got {len(parameters_from_other_nodes)} parameters from other nodes."
            )
            LOGGER.info(
                f"Waiting for {self.num_nodes - 1 - len(parameters_from_other_nodes)} more."
            )
            LOGGER.info(f"Waiting {wait_counter} seconds..")
            time.sleep(min(60 * 2**wait_counter, max_wait))
            wait_counter += 1
            parameters_from_other_nodes = self._get_parameters_from_other_nodes(epoch)

        # Aggregate the parameters from other nodes
        parameters_from_self_and_other_nodes = parameters_from_other_nodes + [
            local_parameters
        ]
        aggregated_parameters = self._aggregate(
            parameters_from_self_and_other_nodes,
            [1] * len(parameters_from_self_and_other_nodes),
        )
        # self.model_store["latest_federated"] = aggregated_parameters
        return aggregated_parameters
