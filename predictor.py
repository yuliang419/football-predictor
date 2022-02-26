import tensorflow as tf
import pandas as pd


def get_model(X_train: pd.DataFrame, random_seed: int = 123) -> tf.keras.Sequential:
    """
    Generate model.
    :param X_train: predictors from train set as dataframe
    :param random_seed: random seed for tf.random
    :return: compiled model
    """
    tf.random.set_seed(random_seed)

    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(X_train)

    input_size = X_train.shape[1]

    model = tf.keras.Sequential(
        [
            norm_layer,
            tf.keras.layers.Dense(32, activation="relu", input_shape=(input_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model
