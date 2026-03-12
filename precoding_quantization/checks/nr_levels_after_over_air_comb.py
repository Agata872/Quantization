import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    M = 2
    b = 1
    nr_symb = 2**(2*b*M)

    # Define the two possible values for the real and imaginary parts
    a = 1
    b = -1

    # Create an array of all possible values
    values = np.array([a, b])

    # Generate all possible combinations for real and imaginary parts
    real_combinations = np.array(np.meshgrid(values, values, values, values)).T.reshape(-1, 4)

    # Separate the combinations into the two complex numbers
    z1_real = real_combinations[:, 0]
    z1_imag = real_combinations[:, 1]
    z2_real = real_combinations[:, 2]
    z2_imag = real_combinations[:, 3]

    # Form the complex numbers
    z1 = z1_real + 1j * z1_imag
    z2 = z2_real + 1j * z2_imag

    # Create the final matrix
    x = np.vstack((z1, z2))

    # channels
    h0 = 2+ 3*1j
    h1 = 5.3 + 1.2*1j

    r = np.zeros((x.shape[-1]), dtype=np.complex64)
    for i in range(x.shape[-1]):
        r[i] = x[0, i] * h0 + x[1, i] * h1

    plt.scatter(np.real(r), np.imag(r))
    plt.show()
    print(r)