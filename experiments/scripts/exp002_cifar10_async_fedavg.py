import wandb

# import set_random_seed
from tensorflow.keras.utils import set_random_seed

from experiments.federated_learning_runner import FederatedLearningRunner

# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    num_nodes = 2
    use_async = True
    federated_type = "concurrent"
    dataset = "cifar10"
    strategy = "fedavg"
    data_split = "random"
    # data_split = "partitioned"

    # base config
    config = {
        "epochs": 2,  # 128,
        "batch_size": 32,
        "steps_per_epoch": 100,
        "lr": 0.001,
        "num_nodes": num_nodes,
        "use_async": use_async,
        "federated_type": federated_type,
        "dataset": dataset,
        "strategy": strategy,
        "data_split": data_split,
        "test_steps": 10,
        "net": "resnet50",
    }

    config_overides = []

    # wandb.init(
    #     project="experiments",
    #     entity="flwr_p2p",
    #     name=f"async_{strategy}_{num_nodes}_nodes_{data_split}_split",
    #     config=config,
    # )
    federated_learning_runner = FederatedLearningRunner(
        config=config,
        tracking=False,
    )
    federated_learning_runner.run()
    # wandb.finish()
