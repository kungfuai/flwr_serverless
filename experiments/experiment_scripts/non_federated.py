import wandb
from tensorflow.keras.utils import set_random_seed

from experiments.non_federated_runner import NonFederatedRunner

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

    num_nodes = 2
    dataset = "mnist"

    wandb.init(
        project="test-project", entity="flwr_p2p", name="non_federal", config=config
    )
    nonfederated_runner = NonFederatedRunner(config, num_nodes, dataset)
    nonfederated_runner.run()
    wandb.finish()
