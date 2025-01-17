import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable

from flax.linen import Module

class ShiftAug(Module):
    """
    Random shift image augmentation in JAX, similar to the PyTorch version.
    We pad the image on each side and then randomly choose a patch corresponding
    to the original image size, effectively shifting the image in both directions.
    """
    pad: int = 3

    @staticmethod
    def pad_and_shift(x: jnp.ndarray, rng: jax.random.PRNGKey, pad: int) -> jnp.ndarray:
        # x is expected to be of shape (N, H, W, C)
        # pad x
        x_padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')
        n, h_original, w_original, c = x.shape

        # Sample random offsets
        max_offset = 2 * pad + 1
        dx = jax.random.randint(rng, shape=(n,), minval=0, maxval=max_offset)
        dy = jax.random.randint(rng, shape=(n,), minval=0, maxval=max_offset)

        # For each image in the batch, slice a subwindow of size (h_original, w_original)
        def shift_single_image(args):
            i, (image, ox, oy) = args
            return image[ox:ox + h_original, oy:oy + w_original, :]

        # vmap across the batch dimension
        shifted = jax.vmap(shift_single_image, in_axes=(0, (0, 0, 0)), out_axes=0)(
            jnp.arange(n), (x_padded, dx, dy)
        )
        return shifted

    def __call__(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        return self.pad_and_shift(x, rng, self.pad)

class TDMPC2SimpleConv(nn.Module):
    """
    A simple convolutional encoder in JAX/Flax, inspired by the TDMPC2 architecture.
    Now optionally applies random shift augmentation, normalizes input images to [-0.5, 0.5],
    then applies 4 convolution layers with ReLU, and finally flattens the output. If
    an additional activation is supplied (act), it is applied at the end.
    """
    num_channels: int
    apply_shift_aug: bool = True
    act: Optional[Callable] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # If shift augmentation is requested and rng is provided
        if self.apply_shift_aug and rng is not None:
            x = ShiftAug(pad=3)(x, rng)

        # Pixel preprocessing: normalize to [-0.5, 0.5]
        x = x / 255.0 - 0.5

        # Conv layers
        x = nn.Conv(features=self.num_channels, kernel_size=(7, 7), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=(1, 1))(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Optional final activation
        if self.act is not None:
            x = self.act(x)

        return x

tdmpc2_simple_conv_configs = {
    "tdmpc2_simple_conv": TDMPC2SimpleConv
} 