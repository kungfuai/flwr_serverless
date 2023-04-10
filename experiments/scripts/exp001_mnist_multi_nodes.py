import wandb

from dataclasses import dataclass
from keras.utils import set_random_seed

from experiments.federated_learning_runner import FederatedLearningRunner


@dataclass
class Config:
    # non shared config parameters
    num_nodes: int
    strategy: str

    # shared config parameters
    use_async: bool = True
    federated_type: str = "concurrent"
    dataset: str = "mnist"
    epochs: int = 100
    batch_size: int = 32
    steps_per_epoch: int = 64
    lr: float = 0.001
    test_steps: int = None
    net: str = "simple"
    data_split: str = "skewed"


# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    config1 = Config(num_nodes=2, strategy="fedavg")
    config2 = Config(num_nodes=3, strategy="fedavg")
    config3 = Config(num_nodes=5, strategy="fedavg")
    config4 = Config(num_nodes=2, strategy="fedavgm")
    config5 = Config(num_nodes=3, strategy="fedavgm")
    config6 = Config(num_nodes=5, strategy="fedavgm")
    config7 = Config(num_nodes=2, strategy="fedadam")
    config8 = Config(num_nodes=3, strategy="fedadam")
    config9 = Config(num_nodes=5, strategy="fedadam")

    configs = [
        config1,
        config2,
        config3,
        config4,
        config5,
        config6,
        config7,
        config8,
        config9,
    ]

    for config in configs:
        if config.use_async:
            str_use_async = "async"
        else:
            str_use_async = "sync"
        wandb.init(
            project="multi-nodes-exp001",
            entity="flwr_p2p",
            name=f"mnist_{str_use_async}_{config.num_nodes}nodes_{config.strategy}_{config.data_split}",
            config=config,
        )
        federated_learning_runner = FederatedLearningRunner(
            config=config,
            tracking=True,
        )
        federated_learning_runner.run()
        wandb.finish()
