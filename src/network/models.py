from .layers import Input


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

    def forward(self, X_true):
        self.input_layer.forward(X_true)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output

    def fit(self, X_true, y_true, *, epochs=1, print_every=1):
        self.accuracy.init(y_true)

        for epoch in range(1, epochs+1):
            y_pred = self.forward(X_true)

            data_loss, regularization_loss = self.loss.calculate(
                y_pred, y_true, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(
                y_pred)
            accuracy = self.accuracy.calculate(predictions, y_true)

            self.backward(y_pred, y_true)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_lr}')

    def backward(self, y_pred, y_true):
        self.loss.backward(y_pred, y_true)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)
        loss = self.loss.calculate(y_pred, y_test)
        predictions = self.output_layer_activation.predictions(
            y_pred)
        accuracy = self.accuracy.calculate(predictions, y_test)

        # Print a summary
        print(f'Testing, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')
