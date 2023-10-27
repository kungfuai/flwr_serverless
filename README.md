A Flower ([flwr](https://flower.dev/)) extension for serverless federated learning.

Technical report (arXiv): [Serverless Federated Learning with `flwr-serverless`](https://arxiv.org/abs/2310.15329).

## Install

```
pip install flwr-serverless

or

pip install git+https://github.com/kungfuai/flwr_serverless.git
```

## Usage for tensorflow

- Step 1: Create federated `Node`s that use a shared folder to exchange model weights and use a federated strategy (`flwr.server.strategy.Strategy`) to control how the weights are aggregated.
- Step 2: Create and configure a callback `FlwrFederatedCallback` and use it in the `keras.Model.fit()`.

```python
# Create a FL Node that has a strategy and a shared folder.
from flwr.server.strategy import FedAvg  # This is a flwr federated strategy.
from flwr_serverless import AsyncFederatedNode, S3Folder
from flwr_serverless.keras import FlwrFederatedCallback

strategy = FedAvg()
shared_folder = S3Folder(directory="mybucket/experiment1")
node = AsyncFederatedNode(strategy=strategy, shared_folder=shared_folder)

# Create a keras Callback with the FL node.
num_examples_per_epoch = steps_per_epoch * batch_size # number of examples used in each epoch
callback = FlwrFederatedCallback(
    node,
    num_examples_per_epoch=num_examples_per_epoch,
    save_model_before_aggregation=False,
    save_model_after_aggregation=False,
)

# Join the federated learning, by fitting the model with the federated callback.
model = keras.Model(...)
model.compile(...)
model.fit(dataset, callbacks=[callback])
```

`flwr_serverless` uses `flwr_serverless.SharedFolder` to save model weights and metrics. The logic folder can be backed by a storage backend like S3.

The asynchronous FL node does not wait to sync with other nodes. It takes the latest
model weights from other nodes and performs the aggregation according to the specified strategy.

### Running experiments

To make it easier to experimemt with different strategies, we provide utility classes like `flwr.keras.example.FederatedLearningTestRun`. This allows you to configure the dataset partition, strategy and concurrency. Please use this as an example to develop your own experiments.

To reproduce some experiments reported in the paper, run

```
python -m experiments.experiment_scripts.exp1_mnist
python -m experiments.experiment_scripts.exp2_cifar10
python -m experiments.experiment_scripts.exp3_wikitext
```

Each of the above experiments run through a grid search over a large hyperparameter space,
with repeated trials using different random seeds. Please edit the script to adjust
the number of trials and the hyperparameter search space.