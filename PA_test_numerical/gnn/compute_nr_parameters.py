import numpy as np

if __name__ == '__main__':
    layers = 8
    nr_features = 256

    trainable_params = 2 * (3 * (nr_features * 2)) + (layers-2) * ((nr_features * nr_features) * 3)

    print(f'{trainable_params=}')
