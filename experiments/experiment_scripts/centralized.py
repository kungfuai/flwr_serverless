import wandb
from tensorflow.keras.utils import set_random_seed

from experiments.centralized_model import Centralized_Model

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

    model = Centralized_Model(config, dataset)

    # federeated run w/ FedAvg
    wandb.init(
        project="test-project", entity="flwr_p2p", name="centralized", config=config
    )
    federated_learning_run = model.train_and_eval()
    wandb.finish()
