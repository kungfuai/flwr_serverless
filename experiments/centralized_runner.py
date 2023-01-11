import numpy as np
from wandb.keras import WandbCallback


from flwr_p2p.keras.example import CreateMnistModel

from experiments.base_experiment_runner import BaseExperimentRunner


class CentralizedRunner(BaseExperimentRunner):
    def __init__(self, config, num_nodes, dataset):
        super().__init__(config, dataset)
        self.num_nodes = 1

    def run(self):
        self.train_and_eval()

    def train_and_eval(self):
        image_size = self.x_train.shape[1]
        x_train = np.reshape(self.x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(self.x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        model = CreateMnistModel(self.lr).run()

        model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=[WandbCallback()],
        )
        # memorization test
        loss, accuracy = model.evaluate(
            x_test, self.y_test, batch_size=self.batch_size, steps=self.steps_per_epoch
        )
