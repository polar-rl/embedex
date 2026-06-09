from abc import ABC, abstractmethod

import jax

from embedex.functions.config import Config
from embedex.functions.embedding import FunctionEmbedding


class Generator(ABC):
    @abstractmethod
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def operator_input_size(self, cfg: Config) -> int:
        """Return the input size for the operator network."""
        raise NotImplementedError

    @abstractmethod
    def operator_input(
        self, embedding_network: FunctionEmbedding, xs: jax.Array, ys: jax.Array, extra: jax.Array
    ) -> jax.Array:
        """Return the input to the operator network."""
        raise NotImplementedError

    @abstractmethod
    def operator_ground_truth(self, xs: jax.Array, ys: jax.Array, extra: jax.Array) -> jax.Array:
        """Return the ground-truth values the operator's output should decode to."""
        raise NotImplementedError
