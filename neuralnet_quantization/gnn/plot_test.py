import tensorflow as tf
import numpy as np



# Generate some random data
x = np.random.rand(100, 1)
y = x * 2 + np.random.randn(100, 1) * 0.1

# Train a model multiple times in a for loop
for i in range(5):
    tf.random.set_seed(42)
    nr_hidden = 1
    if i > 3:
        nr_hidden = 2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(nr_hidden, input_shape=(1,))
    ])
    model.compile(optimizer='sgd', loss='mse')
    model.fit(x, y, epochs=10, verbose=0)
    print("Iteration", i+1, ":", model.predict([[0.5]]))
