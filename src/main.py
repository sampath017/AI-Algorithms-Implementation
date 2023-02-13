def main():
    import nnfs
    from nnfs.datasets import spiral_data

    from network.layers import Dense

    nnfs.init()

    # Data
    X, y = spiral_data(100, 3)

    dense1 = Dense(2, 3)
    dense1.forward(X)
    
    print(dense1.output[:5])


if __name__ == "__main__":
    main()
