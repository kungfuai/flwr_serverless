import wandb

# import set_random_seed
from tensorflow.keras.utils import set_random_seed

from experiments.federated_learning_runner import FederatedLearningRunner

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

    num_nodes = 3
    use_async = True
    federated_type = "sequential"
    dataset = "mnist"
    strategy = "fedavg"

    # federeated run w/ FedAvg
    wandb.init(project="test-project", entity="flwr_p2p", name="federated_avg_asnyc")
    federated_learning_runner = FederatedLearningRunner(
        config=config,
        num_nodes=num_nodes,
        use_async=use_async,
        federated_type=federated_type,
        dataset=dataset,
        strategy=strategy,
    )
    federated_learning_runner.run()
    wandb.finish()
