def main():
    import numpy as np

    import nnfs
    from nnfs.datasets import spiral_data

    from network.layers import Dense
    from network.activations import ReLU, Softmax
    from network.losses import CategoricalCrossentropy

    nnfs.init()

    # Data
    X, y = spiral_data(100, 3)

    dense1 = Dense(2, 3)
    activation1 = ReLU()
    dense2 = Dense(3, 3)
    activation2 = Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss_function = CategoricalCrossentropy()

    print(activation2.output[:5])
    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print(f"acc: {accuracy}")
    print(f"loss: {loss}")


if __name__ == "__main__":
    main()
