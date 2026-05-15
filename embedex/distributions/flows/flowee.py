from abc import abstractmethod
from typing import Self

import distrax
import equinox as eqx
import jax


class Flow(eqx.Module):
    """Abstract base class for invertible generative models.

    Defines the common interface shared by normalizing flows, flow matching,
    and discrete flow matching: a pair of `_forward` / `_inverse` transforms
    plus `encode`, `sample`, `log_prob`, and `loss` built on top of them.
    """

    prior: distrax.Distribution = eqx.field(init=False, static=True, default=None)
    prior_shape: tuple[int, ...] = eqx.field(init=False, default=None)

    def add_prior(self, prior: distrax.Distribution, shape: tuple[int, ...]) -> Self:
        object.__setattr__(self, 'prior', prior)
        object.__setattr__(self, 'prior_shape', shape)
        return self

    @abstractmethod
    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def encode(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        return self._forward(x, 0, sideinfo=sideinfo, params=params, key=key)[0]

    def sample(
        self,
        sideinfo: jax.Array | None,
        params: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        assert self.prior is not None, 'Must call `add_prior` before calling `sample`'

        sample_key, inverse_key = jax.random.split(key, 2)
        z = self.prior.sample(seed=sample_key, sample_shape=self.prior_shape)
        return self._inverse(z, 0, sideinfo=sideinfo, params=params, key=inverse_key)[0]

    def log_prob(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.prior is not None, 'Must call `add_prior` before calling `log_prob`'
        z, ldj = self._forward(x, 0, sideinfo=sideinfo, params=params, key=key)
        log_prob_z = self.prior.log_prob(z).sum()
        return log_prob_z + ldj, log_prob_z, ldj

    def loss(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        log_prob_x, log_prob_z, ldj = self.log_prob(x, sideinfo=sideinfo, params=params, key=key)

        return -log_prob_x, log_prob_z, ldj
