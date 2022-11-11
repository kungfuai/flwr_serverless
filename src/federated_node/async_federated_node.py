from typing import List, Tuple
from uuid import uuid4
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

    """
    def __init__(self, storage_backend, strategy):
        self.node_id = uuid4()
        self.counter = 0
        self.strategy = strategy
        self.model_store = storage_backend
        self.model_store["latest_federated"] = None
        # self.model_store = {
        #     "last_seen_node_id": None,
        #     "latest_federated": None,
        # }
    
    def _get_latest_federated_model(self) -> Parameters:
        return self.model_store.get("latest_federated", None)
    
    def _aggregate(self, local_parameters: Parameters, federated_parameters: Parameters, num_examples: int = None, federated_num_examples: int = None) -> Parameters:
        # if num_examples is None or federated_num_examples is None:
        #     num_examples = 1
        #     federated_num_examples = 1
        # TODO: allow different num_examples
        num_examples = 1
        federated_num_examples = 1

        # Aggregation using the strategy.
        client_0 = client_1 = None
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                client_0,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=local_parameters,
                    num_examples=num_examples,
                    metrics={},
                ),
            ),
            (
                client_1,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=federated_parameters,
                    num_examples=federated_num_examples,
                    metrics={},
                ),
            ),
        ]
        
        aggregated_parameters, _ = self.strategy.aggregate_fit(
            server_round=self.counter+1, results=results, failures=[]
        )
        self.counter += 1
        return aggregated_parameters

    def update_parameters(self, local_parameters, num_examples: int = None):
        latest_federated_parameters = self._get_latest_federated_model()
        if latest_federated_parameters is None:
            self.model_store["latest_federated"] = local_parameters
            return None
        else:
            aggregated_parameters = self._aggregate(local_parameters, latest_federated_parameters)
            # Optional 1: x_t = avg(x_{t-1}, self_parameters)
            # latest_federated_parameters = aggregated_parameters  # this is worse!
            # Optional 2: avg(self_parameters, other_parameters1, other_parameters2, ...)
            latest_federated_parameters = local_parameters  # this is better!
            # TODO: this is for 2 clients only. Test the 3 client case.
            #   For more than 2 clients, we need to store local parameters for each client
            self.model_store["latest_federated"] = latest_federated_parameters
            return aggregated_parameters