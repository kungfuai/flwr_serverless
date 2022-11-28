import logging
from typing import List, Tuple
from uuid import uuid4
import time
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
)
from flwr.server.client_proxy import ClientProxy


LOGGER = logging.getLogger(__name__)


class AsyncFederatedNode:
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

    References:
    - [Semi-Synchronous Federated Learning for Energy-Efficient
    Training and Accelerated Convergence in Cross-Silo Settings](https://arxiv.org/pdf/2102.02849.pdf)"""

    def __init__(self, storage_backend, strategy):
        self.node_id = str(uuid4())
        self.counter = 0
        self.strategy = strategy
        self.model_store = storage_backend
        self.seen_models = set()
        self.sample_sizes_from_other_nodes = {}  # node_id -> num_examples

    def _aggregate(
        self, parameters_list: List[Parameters], num_examples_list: List[int] = None
    ) -> Parameters:
        # TODO: allow different num_examples
        #  if num_examples_list is None:
        num_examples_list = [1] * len(parameters_list)
        # LOGGER.warning(
        #     f"node: {self.node_id[:2]}, num_examples_list: {num_examples_list}"
        # )

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

    def _get_parameters_from_other_nodes(self) -> List[Parameters]:
        unseen_parameters_from_other_nodes = []
        num_examples_from_other_nodes = []
        for key, value in self.model_store.items():
            if key.startswith("accum_num_examples_"):
                continue
            if isinstance(value, dict) and "parameters" in value:
                if key != self.node_id:
                    model_hash = value["model_hash"]
                    if True:  # model_hash not in self.seen_models:
                        self.seen_models.add(model_hash)
                        unseen_parameters_from_other_nodes.append(value["parameters"])
                        # accum_num_examples_for_this_node = self.model_store[
                        #     f"accum_num_examples_{key}"
                        # ]
                        # accum_num_examples_for_this_node = (
                        #     accum_num_examples_for_this_node or 0
                        # )
                        # unseen_num_examples_from_this_node = (
                        #     accum_num_examples_for_this_node
                        #     - self.sample_sizes_from_other_nodes.get(key, 0)
                        # )
                        # self.sample_sizes_from_other_nodes[
                        #     key
                        # ] = accum_num_examples_for_this_node
                        # num_examples_from_other_nodes.append(
                        #     unseen_num_examples_from_this_node
                        # )
        return unseen_parameters_from_other_nodes, 0  # num_examples_from_other_nodes

    def update_parameters(
        self,
        local_parameters: Parameters,
        upload_only: bool = False,
        num_examples: int = None,
    ):
        LOGGER.info(f"node {self.node_id}: in update_parameters")
        self.model_store[self.node_id] = dict(
            parameters=local_parameters, model_hash=self.node_id + str(time.time())
        )
        # sample_size_key = f"accum_num_examples_{self.node_id}"
        # num_examples = num_examples or 0
        # if self.model_store.get(sample_size_key, None) is None:
        #     self.model_store[sample_size_key] = num_examples
        # else:
        #     self.model_store[sample_size_key] += num_examples
        # print(f"\n{len(self.model_store)} nodes\n")
        (
            parameters_from_other_nodes,
            num_examples_from_other_nodes,
        ) = self._get_parameters_from_other_nodes()
        LOGGER.info(
            f"node {self.node_id}: {len(parameters_from_other_nodes or [])} parameters_from_other_nodes"
        )
        if len(parameters_from_other_nodes) == 0:
            # No other nodes, so just return the local parameters
            return local_parameters
        else:
            # Aggregate the parameters from other nodes
            parameters_from_self_and_other_nodes = [
                local_parameters
            ] + parameters_from_other_nodes
            aggregated_parameters = self._aggregate(
                parameters_from_self_and_other_nodes,
                # num_examples_list=[num_examples] + num_examples_from_other_nodes,
            )
            # self.model_store["latest_federated"] = aggregated_parameters
            return aggregated_parameters
