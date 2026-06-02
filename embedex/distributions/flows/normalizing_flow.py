from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from embedex.distributions.flows.flowee import Flow


class Identity(Flow):
    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return x, ldj


class Inverse(Flow):
    flow: Flow

    def __new__(cls, flow: Flow, *args, **kwargs) -> Flow:
        if flow.__class__ == Inverse:
            return flow.flow
        else:
            return super().__new__(cls, *args, **kwargs)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return self.flow._inverse(x, ldj, key=key)

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return self.flow._forward(x, ldj, key=key)


class Reverse(Flow):
    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.transpose(x, jnp.arange(len(x.shape))[::-1])
        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.transpose(x, jnp.arange(len(x.shape))[::-1])
        return x, ldj


class Rescale(Flow):
    alpha: float

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        ldj += jnp.log(self.alpha) * x.size
        x = x * self.alpha
        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        ldj -= jnp.log(self.alpha) * x.size
        x = x / self.alpha
        return x, ldj


class Reshape(Flow):
    in_shape: tuple
    out_shape: tuple

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.reshape(x, self.out_shape)
        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.reshape(x, self.in_shape)
        return x, ldj


class Sigmoid(Flow):
    eps: float = 0.0

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        ldj += (-x + 2 * jax.nn.log_sigmoid(x)).sum()
        x = jax.nn.sigmoid(x)

        if self.eps is not None:
            ldj -= jnp.log(1 - self.eps) * x.size
            x = (x - 0.5 * self.eps) / (1 - self.eps)

        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if self.eps is not None:
            ldj += jnp.log(1 - self.eps) * x.size
            x = x * (1 - self.eps) + 0.5 * self.eps

        ldj -= jnp.log(x - x**2).sum()
        x = jnp.log(x) - jnp.log(1 - x)

        return x, ldj


class NormalizingFlowModel(Flow):
    flows: list[Flow]

    def loss(self, *args, **kwargs) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
        if any(f.prior is not None for f in self.flows):
            return self.residual_loss(*args, **kwargs)
        else:
            return super().loss(*args, **kwargs)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if key is None:
            for f in self.flows:
                x, ldj = f._forward(x, ldj, sideinfo=sideinfo, params=params, key=None)
            return x, ldj

        keys = jax.random.split(key, len(self.flows))
        for f, k in zip(self.flows, keys, strict=True):
            x, ldj = f._forward(x, ldj, sideinfo=sideinfo, params=params, key=k)
        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if key is None:
            for f in self.flows[::-1]:
                x, ldj = f._inverse(x, ldj, sideinfo=sideinfo, params=params, key=None)
            return x, ldj

        keys = jax.random.split(key, len(self.flows))
        for f, k in zip(self.flows[::-1], keys, strict=True):
            x, ldj = f._inverse(x, ldj, sideinfo=sideinfo, params=params, key=k)
        return x, ldj

    def residual_loss(self, x: jax.Array, key: jax.Array | None = None) -> jax.Array:

        rv = 0
        ldj = 0

        if key is None:
            for f in self.flows:
                x, ldj = f._forward(x, ldj, key=None)

                if f.prior is not None:
                    rv += f.prior.log_prob(x).sum() + ldj
                    x = jax.lax.stop_gradient(x)
                    ldj = 0

            if self.prior is not None:
                rv += self.prior.log_prob(x).sum() + ldj

            return -rv

        keys = jax.random.split(key, len(self.flows))
        for f, k in zip(self.flows, keys, strict=True):
            x, ldj = f._forward(x, ldj, key=k)

            if f.prior is not None:
                rv += f.prior.log_prob(x).sum() + ldj
                x = jax.lax.stop_gradient(x)
                ldj = 0

        if self.prior is not None:
            rv += self.prior.log_prob(x).sum() + ldj

        return -rv


class ParameterizedAffine(Flow):
    """One-dimensional affine transformation"""

    params: jax.Array | None
    scale: jax.Array

    def __init__(self, shape: tuple[int, ...], params: jax.Array | None = None) -> None:
        self.params = params
        self.scale = jnp.zeros(shape)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        assert params is not None or self.params is not None, 'No parameters provided'
        params = cast(jax.Array, params if params is not None else self.params)
        a, b = params[:2]

        scale = jax.nn.softplus(self.scale)
        a = jnp.tanh(a / scale) * scale

        x = x * jnp.exp(a) + b
        ldj += a.sum()

        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        assert params is not None or self.params is not None, 'No parameters provided'
        params = cast(jax.Array, params if params is not None else self.params)
        a, b = params[:2]

        scale = jax.nn.softplus(self.scale)
        a = jnp.tanh(a / scale) * scale

        x = (x - b) / jnp.exp(a)
        ldj -= a.sum()

        return x, ldj


class ParameterizedHinge(Flow):
    """One-dimensional invertible hinge transformation"""

    params: jax.Array | None
    scale: jax.Array

    def __init__(self, shape: tuple[int, ...], params: jax.Array | None = None) -> None:
        self.params = params
        self.scale = jnp.zeros((2,) + shape)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        assert params is not None or self.params is not None, 'No parameters provided'
        params = cast(jax.Array, params if params is not None else self.params)

        scale = jax.nn.softplus(self.scale)
        a, b = list(jnp.tanh(params[:2] / scale) * scale)

        ldj += jnp.where(x < 0, a, b).sum()
        x = jnp.where(x < 0, x * jnp.exp(a), x * jnp.exp(b))

        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        assert params is not None or self.params is not None, 'No parameters provided'
        params = cast(jax.Array, params if params is not None else self.params)

        scale = jax.nn.softplus(self.scale)
        a, b = list(jnp.tanh(params[:2] / scale) * scale)

        ldj -= jnp.where(x < 0, a, b).sum()
        x = jnp.where(x < 0, x / jnp.exp(a), x / jnp.exp(b))

        return x, ldj


class ParameterizedNLSq(Flow):
    """One-dimensional non-linear squared transformation

    f(x) = a + bx + c / (1 + (dx + f)^2)

    This is guaranteed to be invertible under certain requirements
    which are guaranteed by the parameter representation.
    """

    params: jax.Array | None
    scale_b: jax.Array
    scale_d: jax.Array

    _log_k = np.log(8 * np.sqrt(3) / 9 * 0.95)

    def __init__(self, shape: tuple[int, ...], params: jax.Array | None = None) -> None:
        self.params = params
        self.scale_b = jnp.zeros(shape)
        self.scale_d = jnp.zeros(shape)

    def _get_params(self, params: jax.Array | None = None) -> tuple[jax.Array, ...]:
        assert params is not None or self.params is not None, 'No parameters provided'
        params = cast(jax.Array, params if params is not None else self.params)
        _a, _b, _c, _d, _f = params[:5]

        scale_b = jax.nn.softplus(self.scale_b)
        scale_d = jax.nn.softplus(self.scale_d)
        _b = jnp.tanh(_b / scale_b) * scale_b
        _d = jnp.tanh(_d / scale_d) * scale_d

        a = _a
        b = jnp.exp(_b)
        d = jnp.exp(_d)
        c = jnp.tanh(_c) * jnp.exp(self._log_k + _b - _d)
        f = _f

        return a, b, c, d, f

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        a, b, c, d, f = self._get_params(params)

        u = d * x + f
        ldj += jnp.log(b - (2 * c * d * u / ((1 + u**2) ** 2))).sum()
        x = a + b * x + c / (1 + u**2)

        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        a, b, c, d, f = self._get_params(params)

        A = -b * d**2
        B = (x - a) * d**2 - 2 * b * d * f
        C = 2 * d * f * (x - a) - b * (f**2 + 1)
        D = (x - a) * (f**2 + 1) - c

        xN = -B / (3 * A)
        yN = A * xN**3 + B * xN**2 + C * xN + D
        deltasq = (B**2 - 3 * A * C) / (9 * A**2)

        delta = jnp.sqrt(jnp.abs(deltasq))
        h = 2 * A * delta**3

        sign = jnp.sign(yN / h)

        real_x = xN - 2 * sign * delta * jnp.cosh(jnp.arccosh(sign * yN / h) / 3)
        imag_x = xN - 2 * delta * jnp.sinh(jnp.arcsinh(yN / h) / 3)
        x = jnp.select([deltasq >= 0, deltasq < 0], [real_x, imag_x])

        u = d * x + f
        ldj -= jnp.log(b - (2 * c * d * u / ((1 + u**2) ** 2))).sum()

        return x, ldj


class Coupling(Flow):
    mask: jax.Array
    transform_param_net: eqx.Module
    transform: Flow | None = None
    dual: bool = False

    def __post_init__(self) -> None:
        if self.transform is None:
            self.transform = ParameterizedAffine(self.mask.shape)

    def _get_params(
        self, x: jax.Array, mask: jax.Array, sideinfo: jax.Array | None = None
    ) -> jax.Array:
        if sideinfo is not None:
            params = self.transform_param_net(x * mask, sideinfo)
        else:
            params = self.transform_param_net(x * mask)

        return params.reshape((-1,) + x.shape) * jnp.expand_dims(1 - mask, 0)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        params = self._get_params(x, self.mask, sideinfo=sideinfo)
        x, ldj = cast(Flow, self.transform)._forward(
            x, ldj, sideinfo=sideinfo, params=params, key=key
        )

        if self.dual:
            params = self._get_params(x, 1 - self.mask, sideinfo=sideinfo)
            x, ldj = cast(Flow, self.transform)._forward(
                x, ldj, sideinfo=sideinfo, params=params, key=key
            )

        return x, ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if self.dual:
            params = self._get_params(x, 1 - self.mask, sideinfo=sideinfo)
            x, ldj = cast(Flow, self.transform)._inverse(
                x, ldj, sideinfo=sideinfo, params=params, key=key
            )

        params = self._get_params(x, self.mask, sideinfo=sideinfo)
        x, ldj = cast(Flow, self.transform)._inverse(
            x, ldj, sideinfo=sideinfo, params=params, key=key
        )

        return x, ldj


def create_mask(shape: tuple[int, ...], block_size: tuple[int, ...]) -> jax.Array:
    """Generates masks for use with Coupling layers.

    shape: shape of the mask
    block_size: size of a constant-mask block

    If block_size[i] is negative, it partitions the i axis into block_size[i]
    number of blocks.

    Examples:
    >>> create_mask((8, 8), (2, 2))
    [[0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 1 0 0]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 1 0 0]]

    >>> create_mask((8, 8), (-1, 2))
    [[0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]]

    >>> create_mask((8, 8), (-2, -4))
    [[0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [0 0 1 1 0 0 1 1]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 1 0 0]
     [1 1 0 0 1 1 0 0]]
    """

    block_size = [shape[i] // -d if d < 0 else d for i, d in enumerate(block_size)]
    mask = (
        sum(
            [
                jnp.reshape(
                    jnp.arange(shape[i]) // block_size[i],
                    [1 if j != i else -1 for j in range(len(shape))],
                )
                for i in range(len(shape))
            ]
        )
        % 2
    )

    return mask
