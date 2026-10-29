"""`GraphEmbedding`: the set encoder over sampled edges and the conditioned ALLO model on top."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.graphs.families.random_trees import RandomTreesGenerator

N = 8  # vertices per graph; matches the `embedding_network` fixture
EMBEDDING_SIZE, NUM_DIMS = 8, 4


@pytest.fixture
def edges(key) -> jax.Array:
    """Edges sampled from two different random trees, shape (2, num_samples, 2, N)."""

    edges, _ = RandomTreesGenerator(N).sample(2, 64, key)
    return edges


def with_betas_bias(network, value: float):
    """The network with its dual variables shifted, e.g. so every eigenvalue goes negative."""

    return eqx.tree_at(lambda m: m.betas.net.layers[-1].bias, network, jnp.full((1,), value))


def test_embedding_is_a_finite_vector(embedding_network, edges):
    z = embedding_network.embed(edges[0])

    assert z.shape == (EMBEDDING_SIZE,)
    assert jnp.all(jnp.isfinite(z))


def test_embedding_ignores_the_order_of_the_edges(embedding_network, edges, key):
    shuffled = jax.random.permutation(key, edges[0])

    np.testing.assert_allclose(
        embedding_network.embed(shuffled), embedding_network.embed(edges[0]), atol=1e-5
    )


def test_embedding_ignores_the_direction_of_each_edge(embedding_network, edges):
    reversed_edges = edges[0][:, ::-1]

    np.testing.assert_allclose(
        embedding_network.embed(reversed_edges), embedding_network.embed(edges[0]), atol=1e-5
    )


def test_different_graphs_get_different_embeddings(embedding_network, edges):
    assert not np.allclose(embedding_network.embed(edges[0]), embedding_network.embed(edges[1]))


def test_betas_are_lower_triangular(embedding_network, edges):
    betas = embedding_network.betas(embedding_network.embed(edges[0]))

    assert betas.shape == (NUM_DIMS, NUM_DIMS)
    np.testing.assert_array_equal(jnp.triu(betas, k=1), 0.0)


def test_eigenvalues_are_minus_half_the_betas_diagonal(embedding_network, edges):
    z = embedding_network.embed(edges[0])

    np.testing.assert_allclose(
        embedding_network.eigenvalues(z), -jnp.diag(embedding_network.betas(z)) / 2
    )


def test_eigenfunctions_give_one_value_per_dimension(embedding_network, edges):
    z = embedding_network.embed(edges[0])
    uvs = jax.vmap(lambda v: embedding_network.eigenfunctions(v, z))(jnp.eye(N))

    assert uvs.shape == (N, NUM_DIMS)


def test_representation_drops_the_constant_eigenfunction(embedding_network, edges):
    z = embedding_network.embed(edges[0])

    assert embedding_network.representation(jnp.eye(N), z).shape == (N, NUM_DIMS - 1)


def test_distances_are_squared_euclidean_distances_of_the_representation(embedding_network, edges):
    z = embedding_network.embed(edges[0])
    rep = embedding_network.representation(jnp.eye(N), z)
    expected = jnp.sum((rep[:, jnp.newaxis] - rep[jnp.newaxis, :]) ** 2, axis=-1)

    distances = embedding_network.distance_matrix(jnp.eye(N), z)

    np.testing.assert_allclose(distances, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(distances, distances.T, atol=1e-5)
    np.testing.assert_allclose(jnp.diag(distances), 0.0, atol=1e-5)


def test_distances_stay_finite_when_eigenvalues_are_not_positive(embedding_network, edges):
    network = with_betas_bias(embedding_network, 10.0)
    z = network.embed(edges[0])

    assert jnp.all(network.eigenvalues(z) < 0)
    assert jnp.all(jnp.isfinite(network.distance_matrix(jnp.eye(N), z)))


def test_laplacian_is_symmetric(embedding_network, edges):
    laplacian = embedding_network.laplacian(jnp.eye(N), embedding_network.embed(edges[0]))

    assert laplacian.shape == (N, N)
    np.testing.assert_allclose(laplacian, laplacian.T, atol=1e-5)


def test_loss_and_its_gradients_are_finite(embedding_network, edges, key):
    loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(edges[0], key))(embedding_network)

    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))


def test_loss_depends_on_how_the_edges_are_split(embedding_network, edges, key):
    # The loss embeds a random half of the edges and scores the other half.
    first = embedding_network.loss(edges[0], key)
    second = embedding_network.loss(edges[0], jax.random.key(1))

    assert not np.isclose(first, second)
