def main():
    import nnfs
    from nnfs.datasets import spiral_data

    from network.layers import Dense
    from network.activations import ReLU, Softmax

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

    print(activation2.output[:5])


if __name__ == "__main__":
    main()
