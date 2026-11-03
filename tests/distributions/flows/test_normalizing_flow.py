"""Normalizing flows: invertible layers with exact log-determinants, and the density they define."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.flows.normalizing_flow import (
    Coupling,
    Inverse,
    NormalizingFlowModel,
    ParameterizedAffine,
    ParameterizedHinge,
    ParameterizedNLSq,
    Rescale,
    Sigmoid,
    create_mask,
)

DIM = 16

# Each elementwise transform and the number of parameters it reads per coordinate.
ELEMENTWISE = {ParameterizedAffine: 2, ParameterizedHinge: 2, ParameterizedNLSq: 5}


def jacobian_log_det(fn, x: jax.Array) -> float:
    return float(jnp.linalg.slogdet(jax.jacfwd(fn)(x))[1])


@pytest.fixture(params=list(ELEMENTWISE), ids=lambda cls: cls.__name__)
def elementwise(request, key):
    """An elementwise transform with random learned scales, and random parameters for it."""

    cls = request.param
    scale_key, params_key = jax.random.split(key)
    transform = cls((DIM,))
    scales = [name for name in ('scale', 'scale_b', 'scale_d') if hasattr(transform, name)]
    for name, k in zip(scales, jax.random.split(scale_key, len(scales)), strict=True):
        value = jax.random.normal(k, getattr(transform, name).shape)
        transform = eqx.tree_at(lambda t, n=name: getattr(t, n), transform, value)

    return transform, 2.0 * jax.random.normal(params_key, (ELEMENTWISE[cls], DIM))


@pytest.fixture
def xs(key) -> jax.Array:
    return 3.0 * jax.random.normal(jax.random.fold_in(key, 1), (DIM,))


class TestElementwiseTransforms:
    def test_inverse_undoes_forward(self, elementwise, xs):
        transform, params = elementwise

        ys, ldj = transform._forward(xs, 0.0, params=params)
        back, inverse_ldj = transform._inverse(ys, 0.0, params=params)

        np.testing.assert_allclose(back, xs, rtol=1e-4, atol=1e-4)
        assert float(inverse_ldj) == pytest.approx(-float(ldj), rel=1e-4, abs=1e-4)

    def test_log_det_is_that_of_the_jacobian(self, elementwise, xs):
        transform, params = elementwise
        jacobian = jax.jacfwd(lambda x: transform._forward(x, 0.0, params=params)[0])(xs)

        _, ldj = transform._forward(xs, 0.0, params=params)

        np.testing.assert_allclose(jacobian, jnp.diag(jnp.diag(jacobian)), atol=1e-6)
        expected = float(jnp.sum(jnp.log(jnp.abs(jnp.diag(jacobian)))))
        assert float(ldj) == pytest.approx(expected, rel=1e-4, abs=1e-4)

    def test_nlsq_is_increasing_even_for_extreme_parameters(self):
        # The 0.95 margin on c keeps f'(x) = b - 2cdu / (1 + u^2)^2 positive for all parameters.
        transform = ParameterizedNLSq((1,))
        extremes = jnp.array([-20.0, 0.0, 20.0])
        grid = jnp.stack(jnp.meshgrid(*[extremes] * 5, indexing='ij'), -1).reshape(-1, 5, 1)
        xs = jnp.linspace(-30.0, 30.0, 601)

        def derivative(params, x):
            return jax.grad(lambda x: transform._forward(x[None], 0.0, params=params)[0][0])(x)

        slopes = jax.vmap(lambda p: jax.vmap(lambda x: derivative(p, x))(xs))(grid)

        assert float(slopes.min()) > 0.0


@pytest.mark.parametrize(
    'flow',
    [Sigmoid(), Rescale(alpha=2.5), Inverse(Sigmoid())],
    ids=['sigmoid', 'rescale', 'inverse'],
)
def test_parameter_free_layers_invert_with_the_jacobian_log_det(flow, key):
    # Inverse(Sigmoid) is the logit, defined on (0, 1).
    xs = jax.random.uniform(key, (DIM,), minval=0.05, maxval=0.95)
    if not isinstance(flow, Inverse):
        xs = 4.0 * xs - 2.0

    ys, ldj = flow._forward(xs, 0.0)
    back, inverse_ldj = flow._inverse(ys, 0.0)

    np.testing.assert_allclose(back, xs, rtol=1e-4, atol=1e-5)
    assert float(inverse_ldj) == pytest.approx(-float(ldj), rel=1e-4)
    assert float(ldj) == pytest.approx(
        jacobian_log_det(lambda x: flow._forward(x, 0.0)[0], xs), rel=1e-4
    )


@pytest.mark.parametrize(
    'shape, block_size, expected',
    [
        ((6,), (1,), [0, 1, 0, 1, 0, 1]),
        ((4, 4), (2, 2), [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]),
        ((4, 4), (-1, 2), [[0, 0, 1, 1]] * 4),
        ((4, 4), (-2, -2), [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]),
    ],
)
def test_masks_alternate_blocks(shape, block_size, expected):
    np.testing.assert_array_equal(create_mask(shape, block_size), expected)


class TestFlowModel:
    """The conditional normalizing flow the builder makes: dual coupling layers of NLSq
    transforms, whose parameters an MLP reads from the other half and the embedding."""

    @pytest.fixture
    def flow(self, make_model) -> NormalizingFlowModel:
        return make_model(flow_type='nf', input_dim=2, conditioning='film').generator

    @pytest.fixture
    def sideinfo(self, key) -> jax.Array:
        return jax.random.normal(jax.random.fold_in(key, 2), (8,))

    def test_inverse_undoes_forward(self, flow, sideinfo, key):
        x = 2.0 * jax.random.normal(key, (2,))

        z, ldj = flow._forward(x, 0.0, sideinfo=sideinfo)
        back, inverse_ldj = flow._inverse(z, 0.0, sideinfo=sideinfo)

        np.testing.assert_allclose(back, x, atol=1e-4)
        assert float(inverse_ldj) == pytest.approx(-float(ldj), rel=1e-4, abs=1e-4)

    def test_log_det_is_that_of_the_jacobian(self, flow, sideinfo, key):
        x = 2.0 * jax.random.normal(key, (2,))

        _, ldj = flow._forward(x, 0.0, sideinfo=sideinfo)

        expected = jacobian_log_det(lambda x: flow.encode(x, sideinfo), x)
        assert float(ldj) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_density_integrates_to_one(self, flow, sideinfo):
        grid = jnp.linspace(-12.0, 12.0, 481)
        points = jnp.stack(jnp.meshgrid(grid, grid), -1).reshape(-1, 2)

        log_probs, _, _ = jax.vmap(lambda x: flow.log_prob(x, sideinfo))(points)

        mass = float(jnp.sum(jnp.exp(log_probs)) * (grid[1] - grid[0]) ** 2)
        assert mass == pytest.approx(1.0, abs=0.01)

    def test_samples_encode_to_the_standard_normal_prior(self, flow, sideinfo, key):
        samples = jax.vmap(lambda k: flow.sample(sideinfo, None, k))(jax.random.split(key, 4096))

        z = jax.vmap(lambda x: flow.encode(x, sideinfo))(samples)

        np.testing.assert_allclose(z.mean(0), 0.0, atol=0.06)
        np.testing.assert_allclose(z.std(0), 1.0, atol=0.05)

    def test_the_embedding_changes_the_density(self, flow, sideinfo, key):
        x = jax.random.normal(key, (2,))

        assert flow.log_prob(x, sideinfo)[0] != flow.log_prob(x, -sideinfo)[0]


def test_one_sided_coupling_passes_the_conditioning_half_through(make_model, key):
    coupling = make_model(flow_type='nf', input_dim=4).generator.flows[0]
    one_sided = Coupling(coupling.mask, coupling.transform_param_net, coupling.transform)
    x = jax.random.normal(key, (4,))
    sideinfo = jax.random.normal(jax.random.fold_in(key, 1), (8,))

    y, _ = one_sided._forward(x, 0.0, sideinfo=sideinfo)

    kept = coupling.mask == 1
    np.testing.assert_array_equal(y[kept], x[kept])
    assert not np.allclose(y[~kept], x[~kept])
