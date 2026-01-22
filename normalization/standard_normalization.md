---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 1. Normalization



# 2. Practice

```python
import numpy as np
import matplotlib.pyplot as plt

# Define sample linear function
alpha = 3
beta = -3

# Set the random seed
rng = np.random.default_rng(0)

# Generate noisy sample data
x = np.linspace(-10, 10, 100)
norm_x = (x - x.mean()) / x.std()
delta = rng.uniform(low=-7, high=7, size=100)
y = alpha * norm_x + beta # + delta

def cost_fn(w1, w2):
  y_pred = w1 * norm_x + w2
  return np.mean((y_pred - y)**2)

# define a grid of weights that encapsulate our sample function
w1_vals = np.linspace(-5, 5, 100)
w2_vals = np.linspace(-5, 5, 100)

W1, W2 = np.meshgrid(w1_vals, w2_vals)

J = np.zeros(W1.shape)

for i in range(W1.shape[0]):
  for j in range(W1.shape[1]):
    J[i,j] = cost_fn(W1[i,j], W2[i,j])

min_J = J.min()
levels = np.linspace(min_J, min_J + 100, 20)
plt.contour(W1, W2, J, levels=levels)
plt.xlabel("w1")
plt.ylabel("w2")
plt.colorbar(label='Cost J')
plt.show()

```
