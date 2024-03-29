import flax.linen as nn
from jax import stop_gradient


class Classifier(nn.Module):
    roberta: nn.Module

    def __call__(self, x):
        x = self.roberta(x)
        stop_gradient(x)
        nn.Dense(features=2)(x)
        x = nn.softmax(x)
        return x