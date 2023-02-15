from tensorflow import keras
import keras_cv


class ResNetModelBuilder:
    def __init__(
        self,
        lr=0.001,
        include_rescaling=False,
        num_classes=10,
        weights=None,
        net="ResNet18",
        input_shape=None,
    ):
        self.lr = lr
        self.num_classes = num_classes
        self.weights = weights
        self.net = net
        self.input_shape = input_shape
        self.include_rescaling = include_rescaling

    def run(self):
        if self.net == "ResNet50":
            backbone = keras_cv.models.ResNet50(
                include_rescaling=self.include_rescaling,
                # include_top=True,
                include_top=False,
                weights=self.weights,
                classes=self.num_classes,
                input_shape=(None, None, 3),
            )
            x = keras.layers.Input(shape=(None, None, 3))
            y = backbone(x)
            y = keras.layers.GlobalAveragePooling2D()(y)
            y = keras.layers.Dense(self.num_classes, activation="softmax")(y)
            model = keras.Model(x, y)
            # model = keras.applications.resnet50.ResNet50(
            #     include_top=True,
            #     weights=self.weights,
            #     classes=self.num_classes,
            #     input_shape=(None, None, 3),
            # )
        else:
            fn = getattr(keras_cv.models, self.net)
            model = fn(
                include_rescaling=self.include_rescaling,
                include_top=True,
                weights=self.weights,
                classes=self.num_classes,
                input_shape=(None, None, 3),
            )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.lr),
            metrics=["accuracy"],
        )
        return model
