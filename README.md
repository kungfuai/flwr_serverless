A Flower ([flwr](https://flower.dev/)) extension, to enable peer-to-peer federated learning.

## Usage for tensorflow

- Step 1: Create federated `Node`s that use a shared folder to exchange model weights and use a federated strategy (`flwr.server.strategy.Strategy`) to control how the weights are aggregated.
- Step 2: Create and configure a callback `FlwrFederatedCallback` and use it in the `keras.Model.fit()`.

```python
# Create a FL Node that has a strategy and a shared folder.
from flwr.server.strategy import FedAvg
from flwr_p2p import AsyncFederatedNode, S3Folder

strategy = FedAvg()
shared_folder = S3Folder(directory="mybucket/experiment1")
node = AsyncFederatedNode(strategy=strategy, shared_folder=shared_folder)

# Create a keras Callback with the FL node.
from flwr.keras import FlwrFederatedCallback
num_examples_per_epoch = steps_per_epoch * batch_size # number of examples used in each epoch
callback = FlwrFederatedCallback(
    node,
    num_examples_per_epoch=num_examples_per_epoch,
)

# Join the federated learning, by fitting the model with the federated callback.
model = keras.Model(...)
model.compile(...)
model.fit(dataset, callbacks=[callback])
```

`flwr_p2p` uses `flwr_p2p.SharedFolder` to save states. `flwr_p2p.Folder` is a logical "folder" to hold model checkpoints from FL participants (nodes), as well as model parameters produced by the FL strategy. The logic folder can be backed by a storage backend like S3 or mlflow artifacts (which can in turn be backed by S3).

The asynchronous FL node does not wait to sync with other nodes. It takes the latest
checkpoints from other nodes and performs the aggregation according to the specified strategy.

### Experiment with different strategies

To make it easier to experimemt with different strategies, we provide utility classes like `flwr.keras.example.FederatedLearningTestRun`. This allows you to configure the dataset partition, strategy and concurrency. Please use this as an example to develop your own experiments.
