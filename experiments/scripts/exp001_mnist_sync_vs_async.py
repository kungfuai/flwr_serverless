import wandb

# import set_random_seed
from dataclasses import dataclass
from tensorflow.keras.utils import set_random_seed
from experiments.federated_learning_runner import FederatedLearningRunner


@dataclass
class Config:
    # non shared config parameters
    use_async: bool
    data_split: str

    # shared config parameters
    dataset: str = "mnist"
    epochs: int = 512
    batch_size: int = 32
    steps_per_epoch: int = 64
    lr: float = 0.001
    test_steps: int = None
    net: str = "simple"
    num_nodes: int = 2
    strategy: str = "fedavg"
    federated_type: str = "concurrent"
    skew_factor: float = None


# main function
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    set_random_seed(117)

    # experiment configs
    skew_factor = 0.95
    config1 = Config(use_async=True, data_split="random")
    config2 = Config(use_async=True, data_split="partitioned")
    config3 = Config(use_async=True, data_split="skewed", skew_factor=skew_factor)
    config4 = Config(use_async=False, data_split="random")
    config5 = Config(use_async=False, data_split="partitioned")
    config6 = Config(use_async=False, data_split="skewed", skew_factor=skew_factor)

    configs = [config1, config2, config3, config4, config5, config6]

    # run experiments
    for config in configs:
        # print(os.getenv("WANDB_PROJECT"))
        if config.use_async:
            use_async = "async"
        else:
            use_async = "sync"

        name = f"mnist_{use_async}_{config.data_split}_split"

        if config.data_split == "skewed":
            name += f"_{config.skew_factor}"

        wandb.init(
            project="sync-vs-async",
            entity="flwr_p2p",
            name=name,
            config=config,
        )
        federated_learning_runner = FederatedLearningRunner(
            config=config,
            tracking=True,
        )
        federated_learning_runner.run()
        wandb.finish()
