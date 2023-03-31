import wandb

from keras.utils import set_random_seed

from experiments.federated_learning_runner import FederatedLearningRunner

# main function
if __name__ == "__main__":
    # starts a new run
    set_random_seed(117)

    # shared config parameters
    federated_type = "concurrent"
    dataset = "mnist"
    epochs = 100
    batch_size = 32
    steps_per_epoch = 64
    lr = 0.001
    test_steps = None  # uses all test data if None
    net = "simple"
    data_split = "random"

    num_nodes_1 = 2
    fed_strat_1 = "fedavg"
    config1 = {
        "num_nodes": num_nodes_1,
        "strategy": fed_strat_1,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }

    num_nodes_2 = 3
    fed_strat2 = "fedavg"
    config2 = {
        "num_nodes": num_nodes_2,
        "strategy": fed_strat2,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
    num_nodes_3 = 5
    fed_strat_3 = "fedavg"
    config3 = {
        "num_nodes": num_nodes_3,
        "strategy": fed_strat_3,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }

    num_nodes_4 = 2
    fed_strat_4 = "fedavgm"
    config4 = {
        "num_nodes": num_nodes_4,
        "strategy": fed_strat_4,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
    num_nodes_5 = 3
    fed_strat_5 = "fedavgm"
    config5 = {
        "num_nodes": num_nodes_5,
        "strategy": fed_strat_5,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
    num_nodes_6 = 5
    fed_strat_6 = "fedavgm"
    config6 = {
        "num_nodes": num_nodes_6,
        "strategy": fed_strat_6,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }

    num_nodes_7 = 2
    fed_strat_7 = "fedadam"
    config7 = {
        "num_nodes": num_nodes_7,
        "strategy": fed_strat_7,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
    num_nodes_8 = 3
    fed_strat_8 = "fedadam"
    config8 = {
        "num_nodes": num_nodes_8,
        "strategy": fed_strat_8,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
    num_nodes_9 = 5
    fed_strat_9 = "fedadam"
    config9 = {
        "num_nodes": num_nodes_9,
        "strategy": fed_strat_9,
        "use_async": True,
        "data_split": data_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "federated_type": federated_type,
        "dataset": dataset,
        "test_steps": test_steps,
        "net": net,
    }
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
        if config["use_async"]:
            str_use_async = "async"
        else:
            str_use_async = "sync"
        data_split = config["data_split"]
        wandb.init(
            project="multi-nodes-exp001",
            entity="flwr_p2p",
            name=f"mnist_{str_use_async}_{config['num_nodes']}nodes_{config['strategy']}_{data_split}",
            config=config,
        )
        federated_learning_runner = FederatedLearningRunner(
            config=config,
            tracking=True,
        )
        federated_learning_runner.run()
        wandb.finish()
