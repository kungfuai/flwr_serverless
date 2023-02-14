import warnings

warnings.filterwarnings(
    "ignore",
)
import numpy as np
from flwr.server.strategy import FedAvg
from uuid import uuid4
from flwr_p2p.shared_folder.in_memory_folder import InMemoryFolder
from flwr_p2p.keras.example import (
    FederatedLearningTestRun,
)
from experiments.model.keras_models import ResNetModelBuilder


# This test is slow on cpu.
def test_mnist_resnet18_federated_callback_2nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
        model_builder_fn=ResNetModelBuilder(
            num_classes=10,
            lr=0.001,
            net="ResNet18",
        ).run,
        replicate_num_channels=True,
        storage_backend=InMemoryFolder(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05
