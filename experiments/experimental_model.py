class ExperimentModel:
    def __init__(self, config, dataset):
        self.config = config
        if dataset == "mnist":
            from tensorflow.keras.datasets import mnist

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # elif dataset == "cifar10":
        #     from tensorflow.keras.datasets import cifar10
        #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
