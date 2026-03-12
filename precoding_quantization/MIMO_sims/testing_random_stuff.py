import numpy as np

#testing quantization
b = 2
L = 2 ** b  # nr of quantization labels
delta = 1  # optimal step size todo: replace by opimal value
i = np.arange(L)
print(f'{i=}')
labels = delta * (i - (L - 1) / 2)  # todo after quantization, multiply with alpha to respect the avg pwr constraint
print(f'{labels=}')
i = np.arange(1, L)
print(f'inew: {i}')
thresholds = delta * (i - (L / 2))  # thresholds

x = np.array([-2, -1.0, -0.5, 0.5, 2, 16, 0])
#x = np.array([-1, 0, 1])
print(f'{x=}')
print(f'{thresholds=}')
inds = np.digitize(x, thresholds, right=False)
print(inds)
quantized_values = labels[inds]
print(f'{quantized_values=}')