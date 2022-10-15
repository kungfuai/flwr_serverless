A Flower ([flwr](https://flower.dev/)) extension, to enable peer-to-peer federated learning.

## Usage for tensorflow

- Step 1: Create and configure a callback `FlwrFederatedCallback` and use it in the `keras.Model.fit()`.
- Step 2: no step 2.

## How it works

`flwr_p2p` implements peer2peer variations of `flwr.server.strategy.Strategy`. 

```
class FlwrFederatedCallback(keras.callbacks.Callback):
    
    def __init__(self, strategy: flwr_p2p.P2PStrategy, **kwargs):
        self.strategy = strategy
        ...

    def model_to_flwr_parameters(self, model: keras.Model):
        ...

    def update_model_with_flwr_parameters(self, model: keras.Model, parameters):
        ...
    
    def on_epoch_end(self, ...):
        # use the P2PStrategy to update the model.
        # see https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py
        self_parameters = self.model_to_flwr_parameters(self.model)
        fed_parameters = self.strategy.get_current_parameters()
        new_parameters = self.strategy.aggregated_fit([fed_parameters, new_parameters])
        self.model = self.update_model_with_flwr_parameters(self.model, new_parameters)

```

`flwr_p2p.P2PStrategy` uses `flwr_p2p.Folder` to save states. `flwr_p2p.Folder` is a logical "folder" to hold model checkpoints from FL participants, as well as model parameters produced by the FL strategy. The logic folder can be backed by a storage backend like S3 or mlflow artifacts (which can in turn be backed by S3).


