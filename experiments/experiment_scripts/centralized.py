import wandb
from tensorflow.keras.utils import set_random_seed

from experiments.centralized_runner import CentralizedRunner

# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    config = {
        "epochs": 16,
        "batch_size": 32,
        "steps_per_epoch": 8,
        "lr": 0.001,
    }

    dataset = "mnist"

    # federeated run w/ FedAvg
    wandb.init(
        project="test-project", entity="flwr_p2p", name="centralized", config=config
    )
    centralized_runner = CentralizedRunner(config, dataset)
    centralized_runner.run()
    wandb.finish()
