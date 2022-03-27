import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class Predictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_train_val(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        """
        Initialize predictor with training and validation data
        :param X_train: predictors from train set as dataframe
        :param y_train: one-hot encoded labels from train set
        :param X_val: predictors from val set as dataframe
        :param y_val: one-hot encoded labels from val set
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def get_model(self, random_seed: int = 123) -> tf.keras.Sequential:
        """
        Create and initialize model.
        :param random_seed: random seed for tf.random
        """
        tf.random.set_seed(random_seed)

        norm_layer = tf.keras.layers.Normalization()
        norm_layer.adapt(self.X_train)

        input_size = self.X_train.shape[1]

        self.model = tf.keras.Sequential(
            [
                norm_layer,
                tf.keras.layers.Dense(16, activation='relu', input_shape=(input_size,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(3, activation='softmax'),
            ]
        )

        sgd = tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.8, decay=1e-5, nesterov=True)

        self.model.compile(optimizer=sgd, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    def train(self, plot: bool = False, save_model: bool = False, model_name: str = 'model'):
        """
        Train model with given train and val data. Optionally plot metrics and/or save trained model.
        :param plot: if True, show plot of train/val losses and accuracies.
        :param save_model: If true, save trained model
        :param model_name: Name of the saved model
        :return:
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                    mode='min', baseline=None, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), batch_size=32,
                                 epochs=100, callbacks=[callback])
        self.is_trained = True

        if plot:
            ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
            ax.grid()
            _ = ax.set(title='Training and validation loss and accuracy', xlabel='Epochs')
            _ = ax.legend(['Train loss', 'Train accuracy', 'Val loss', 'Val accuracy'])
            plt.show()

        if save_model:
            self.model.save(model_name)
            print(f'Model saved in {model_name}')

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[float, float]:
        """
        Evaluate trained model on test set.
        :param X_test: predictors from test set as dataframe
        :param y_test: one-hot encoded labels from test set
        :return: loss and accuracy on test set
        """
        assert self.is_trained, 'Model has not yet been trained. Train model before evaluating.'
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=32)
        return loss, accuracy

    def load_saved_model(self, model_path: str):
        """
        Load a saved model.
        :param model_path: path to trained model
        :return:
        """
        self.model = tf.keras.models.load_model(model_path)
        self.is_trained = True

    def predict(self, sample: pd.DataFrame) -> pd.DataFrame:
        """
        Use a trained model to make a prediction on a new sample.
        :param sample: one or more samples in a pandas DataFrame on which to perform the prediction. Must be in the same
        format as samples from the test set. The following columns are required:
            'StandingDiff', 'HomeWins', 'AwayWins', 'HomeDraws', 'AwayDraws', 'AvgHomeGoals', 'AvgAwayGoals',
            'AvgHomeShots', 'AvgAwayShots', 'AvgHomeShotsOnTarget', 'AvgAwayShotsOnTarget', 'AvgHomeGoalsConceded',
            'AvgAwayGoalsConceded', 'AvgHomeShotsConceded', 'AvgAwayShotsConceded'
        :return: predicted probabilities as dataframe
        """
        assert self.is_trained, 'Model has not yet been trained. Train model before predicting.'
        required_cols = {'StandingDiff', 'HomeWins', 'AwayWins', 'HomeDraws', 'AwayDraws', 'AvgHomeGoals',
                         'AvgAwayGoals', 'AvgHomeShots', 'AvgAwayShots', 'AvgHomeShotsOnTarget', 'AvgAwayShotsOnTarget',
                         'AvgHomeGoalsConceded', 'AvgAwayGoalsConceded', 'AvgHomeShotsConceded', 'AvgAwayShotsConceded'}

        assert required_cols.issubset(sample.columns), 'Sample must have the following required features: ' + \
                                                       ', '.join(required_cols)

        results = self.model.predict(sample[required_cols])
        return pd.DataFrame(results, columns=['H', 'D', 'A'])
