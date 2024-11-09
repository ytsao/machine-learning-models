from flax import linen as nn  # Linen API
import jax
import jax.numpy as jnp  # JAX NumPy

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax  # Common loss functions and optimizers


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


cnn = CNN()
print(
    cnn.tabulate(
        jax.random.key(0),
        jnp.ones((1, 28, 28, 1)),
        compute_flops=True,
        compute_vjp_flops=True,
    )
)