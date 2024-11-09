from flax import linen as nn 
import jax 
import jax.numpy as jnp 

from clu import metrics
from flax.training import train_state
from flax import struct
import optax

class LeNet(nn.Module):
    """ LeNet model """
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1)) # flatten
        x = nn.Dense(features=256)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=10)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=10)(x)
        return x

