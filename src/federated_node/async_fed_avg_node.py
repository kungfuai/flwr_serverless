import logging
from typing import List, Tuple
from uuid import uuid4
import time
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


LOGGER = logging.getLogger(__name__)


class AsyncFedAvgNode:
    """Async FedAvg with custom implementation rather than
    using an existing flwr strategy.

    Reference: https://arxiv.org/pdf/2102.02849.pdf Algorithm 1.
    """

    def __init__(self, storage_backend):
        self.node_id = str(uuid4())
        self.counter = 0
        self.model_store = storage_backend
        self.seen_models = set()
        self._aggregate_sample_size = 0

    def _get_latest_fed_parameters(self) -> Parameters:
        latest_fed_parameters = self.model_store.get("latest_federated")
        if latest_fed_parameters:
            return latest_fed_parameters
        else:
            return None

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
        if upload_only:
            return None
        # print(f"\n{len(self.model_store)} nodes\n")
        latest_params = self._get_latest_fed_parameters()
        if latest_params is None:
            self.model_store["latest_federated"] = local_parameters
            self.model_store["aggregate_sample_size"] = self.model_store[
                "aggregate_sample_size"
            ] + float(num_examples)
            return local_parameters
        else:
            # Aggregate the parameters from other nodes
            latest_params_np = parameters_to_ndarrays(latest_params)
            # self.model_store["latest_federated"] = aggregated_parameters
            return aggregated_parameters
