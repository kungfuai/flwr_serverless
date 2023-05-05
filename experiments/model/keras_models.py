from tensorflow import keras
import keras_cv


class ResNetModelBuilder:
    def __init__(
        self,
        lr=0.001,
        include_rescaling=False,
        num_classes=10,
        weights=None,
        net="ResNet50",
        input_shape=(None, None, 3),
    ):
        self.lr = lr
        self.num_classes = num_classes
        self.weights = weights
        self.net = net
        self.input_shape = input_shape
        self.include_rescaling = include_rescaling

    def run(self):
        if self.net in ["ResNet50"]:
            fn = getattr(keras_cv.models, self.net)
            backbone = fn(
                include_rescaling=self.include_rescaling,
                include_top=False,
                weights=self.weights,
                classes=self.num_classes,
                input_shape=self.input_shape,
            )
            x = keras.layers.Input(shape=self.input_shape)
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
                input_shape=self.input_shape,
            )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(self.lr),
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    import numpy as np

    model = ResNetModelBuilder(net="ResNet50").run()
    example_input = np.random.rand(2, 32, 32, 3)  # 2 images, 32x32 pixels, 3 channels
    out = model(example_input)
    print("output tensor shape:", out.shape)
