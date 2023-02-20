from .layers import Input
from .losses import CategoricalCrossentropy
from .activations import Softmax
from .commons import ActivationSoftmaxLossCategoricalCrossentropy
import numpy as np


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

        self.input_layer = Input()

        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def forward(self, X_true, training):
        self.input_layer.forward(X_true, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def fit(self, X_true, y_true, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y_true)

        train_steps = 1

        if validation_data:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size:
            train_steps = np.ceil(
                X_true.shape[0] / batch_size).astype(np.int32)

            if validation_data:
                validation_steps = np.ceil(
                    X_val.shape[0] / batch_size).astype(np.int32)

        for epoch in range(1, epochs+1):
            print(f"epoch: {epoch}")

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X_true
                    batch_y = y_true

                else:
                    batch_X = X_true[step*batch_size:(step+1)*batch_size]
                    batch_y = y_true[step*batch_size:(step+1)*batch_size]

                y_pred = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(
                    y_pred, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(
                    y_pred)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(y_pred, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_lr}")

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(
                include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if not epoch % print_every:
                print(
                    f"Training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), lr: {self.optimizer.current_lr}")

            if validation_data:
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]

                    y_pred = self.forward(batch_X, training=False)
                    self.loss.calculate(y_pred, batch_y)

                    predictions = self.output_layer_activation.predictions(
                        y_pred)
                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                print(
                    f"Validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}")

    def backward(self, y_pred, y_true):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(y_pred, y_true)
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(y_pred, y_true)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_test, y_test, *, batch_size=None):
        testing_steps = 1
        if batch_size:
            testing_steps = np.ceil(
                X_test.shape[0] / batch_size).astype(np.int32)

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(testing_steps):
            if batch_size is None:
                batch_X = X_test
                batch_y = y_test
            else:
                batch_X = X_test[step*batch_size:(step+1)*batch_size]
                batch_y = y_test[step*batch_size:(step+1)*batch_size]

            y_pred = self.forward(batch_X, training=False)
            self.loss.calculate(y_pred, batch_y)

            predictions = self.output_layer_activation.predictions(
                y_pred)
            self.accuracy.calculate(predictions, batch_y)

        testing_loss = self.loss.calculate_accumulated()
        testing_accuracy = self.accuracy.calculate_accumulated()

        print(
            f"Testing, acc: {testing_accuracy:.3f}, loss: {testing_loss:.3f}")
