from tensorflow.keras.utils import set_random_seed
from experiments.utils.federated_learning_runner import FederatedLearningRunner

# main function
if __name__ == "__main__":
    # starts a new run
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run federated learning experiments on CIFAR10."
    )

    # base config
    base_config = {
        "project": "cifar10",
        "epochs": 50,
        "batch_size": 16,
        "steps_per_epoch": 300,
        "lr": 0.001,
        "num_nodes": 2,
        "use_async": False,
        "federated_type": "concurrent",
        "dataset": "cifar10",
        "strategy": "fedavg",
        "data_split": "random",
        "skew_factor": 0.9,
        "test_steps": 50,
        "net": "resnet50",
        "track": False,
    }
    for key, value in base_config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.add_argument(
        "--use_default_configs", "-u", action="store_true", default=False
    )

    args = parser.parse_args()
    if args.use_default_configs:
        # Treatments
        config_overides = [
            {
                "use_async": True,
            },
            {
                "use_async": False,
            },
            {
                "use_async": True,
                "data_split": "skewed",
                "skew_factor": 0.9,
            },
            {
                "use_async": True,
                "data_split": "skewed",
                "skew_factor": 0.5,
            },
            {
                "use_async": True,
                "data_split": "skewed",
                "skew_factor": 0.1,
            },
            {
                "use_async": True,
                "num_nodes": 3,
            },
            {
                "use_async": True,
                "num_nodes": 5,
            },
            {
                "use_async": True,
                "data_split": "partitioned",
            },
        ]
        for c in config_overides:
            c["track"] = True
    else:
        config_overide = {}
        for key, value in vars(args).items():
            config_overide[key] = value
        config_overides = [config_overide]

    for i, config_overide in enumerate(config_overides):
        config = {**base_config, **config_overide}
        print(
            f"\n***** Starting trial {i + 1} of {len(config_overides)} with config: {str(config)[:80]}...\n"
        )
        set_random_seed(0)
        federated_learning_runner = FederatedLearningRunner(
            config=config,
            tracking=config["track"],
        )
        federated_learning_runner.run()
