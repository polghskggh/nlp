import flax.linen as nn
from jax import stop_gradient


class Generator(nn.Module):
    roberta: nn.Module

    def __call__(self, x):
        x = self.roberta(x)
        return x