import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.graphs.embedding import GraphEmbedding


class MediansNetwork(eqx.Module):
    net: eqx.nn.MLP

    def __call__(self, vertex: jax.Array, z: jax.Array) -> jax.Array:
        return jax.nn.softplus(self.net(jnp.concatenate([vertex, z]))[0])


class TransitionNetwork(eqx.Module):
    net: eqx.nn.MLP

    def logits(self, distances: jax.Array, z: jax.Array) -> jax.Array:
        n = distances.shape[0]

        z_grid = jnp.broadcast_to(z, (n, n, z.shape[0]))
        inputs = jnp.concatenate([distances[:, :, jnp.newaxis], z_grid], axis=-1)
        scores = jax.vmap(jax.vmap(self.net))(inputs)[..., 0]

        return jnp.where(jnp.eye(n, dtype=jnp.bool_), -jnp.inf, scores)

    def __call__(self, distances: jax.Array, z: jax.Array) -> jax.Array:
        logits = self.logits(distances, z)
        return jax.nn.softmax(logits.reshape(-1)).reshape(logits.shape)


def medians_loss(
    medians_network: MediansNetwork,
    embedding_network: GraphEmbedding,
    edges: jax.Array,
    key: jax.Array,
) -> jax.Array:
    def _per_graph(edges, key):
        perm = jax.random.permutation(key, edges.shape[0])
        edges = edges[perm]

        edges_train, edges_test = jnp.split(edges, 2, axis=0)
        z = embedding_network.embed(edges_train)
        uvs_test = jax.vmap(jax.vmap(lambda v: embedding_network.eigenfunctions(v, z)))(edges_test)

        eig_vals = jnp.clip(embedding_network.eigenvalues(z), 1e-3)
        rep_u = uvs_test[:, 0, 1:] / jnp.sqrt(eig_vals)[jnp.newaxis, 1:]
        rep_v = uvs_test[:, 1, 1:] / jnp.sqrt(eig_vals)[jnp.newaxis, 1:]

        dist = jnp.sum((rep_u - rep_v) ** 2, axis=1)

        med_u = jax.vmap(lambda v: medians_network(v, z))(edges_test[:, 0])
        med_v = jax.vmap(lambda v: medians_network(v, z))(edges_test[:, 1])

        return jnp.mean(jnp.abs(dist - med_u) + jnp.abs(dist - med_v))

    return jnp.mean(jax.vmap(_per_graph)(edges, jax.random.split(key, edges.shape[0])))


def transition_distances(embedding_network: GraphEmbedding, z: jax.Array, n: int) -> jax.Array:
    uvs = jax.vmap(lambda v: embedding_network.eigenfunctions(v, z))(jnp.eye(n))
    rep = uvs[:, 1:] / jnp.sqrt(jnp.clip(embedding_network.eigenvalues(z), 1e-3))[jnp.newaxis, 1:]
    gram = rep @ jnp.transpose(rep)

    return jnp.diag(gram)[:, jnp.newaxis] + jnp.diag(gram)[jnp.newaxis, :] - 2.0 * gram


def transition_loss(
    transition_network: TransitionNetwork,
    embedding_network: GraphEmbedding,
    edges: jax.Array,
    key: jax.Array,
) -> jax.Array:
    def _per_graph(edges, key):
        perm = jax.random.permutation(key, edges.shape[0])
        edges = edges[perm]

        edges_train, edges_test = jnp.split(edges, 2, axis=0)
        z = embedding_network.embed(edges_train)
        distances = transition_distances(embedding_network, z, edges.shape[-1])

        logits = transition_network.logits(distances, z)
        log_probs = jax.nn.log_softmax(logits.reshape(-1)).reshape(logits.shape)
        sources = jnp.argmax(edges_test[:, 0], axis=-1)
        targets = jnp.argmax(edges_test[:, 1], axis=-1)

        return -jnp.mean(log_probs[sources, targets])

    return jnp.mean(jax.vmap(_per_graph)(edges, jax.random.split(key, edges.shape[0])))


def recon_vertex_degrees(distances: jax.Array, degrees: jax.Array) -> jax.Array:
    """Connect each vertex to its `degree`-nearest neighbors, thresholding at
    the midpoint between the degree-th and (degree + 1)-th nearest distances."""

    num_vertices = distances.shape[0]
    degrees = jnp.clip(degrees, 0, num_vertices - 1)

    distances = jnp.where(~jnp.eye(num_vertices, dtype=jnp.bool_), distances, jnp.inf)
    sorted_distances = jnp.sort(distances, axis=1)  # ascending per row; last col is inf (self)

    low = jnp.take_along_axis(
        sorted_distances, jnp.clip(degrees - 1, 0, num_vertices - 1)[:, jnp.newaxis], axis=1
    )[:, 0]
    high = jnp.take_along_axis(sorted_distances, degrees[:, jnp.newaxis], axis=1)[:, 0]

    thresholds = jnp.where(degrees == 0, -jnp.inf, (low + high) / 2.0)
    thresholds = (thresholds[:, jnp.newaxis] + thresholds[jnp.newaxis, :]) / 2.0

    return distances <= thresholds


def recon_average_degree(distances: jax.Array, avg_degree: jax.Array) -> jax.Array:
    """Threshold at the (avg_degree * num_vertices)-th smallest directed distance,
    i.e. predict ~ avg_degree * num_vertices / 2 undirected edges across the graph."""

    num_vertices = distances.shape[0]

    distances = jnp.where(~jnp.eye(num_vertices, dtype=jnp.bool_), distances, jnp.inf)
    cutoff = jnp.clip(
        (avg_degree * num_vertices).astype(jnp.int32), 1, num_vertices * (num_vertices - 1)
    )
    threshold = jnp.sort(distances.ravel())[cutoff - 1]

    return distances <= threshold


def recon_medians(distances: jax.Array, medians: jax.Array) -> jax.Array:
    """Predict an edge iff dist(u, v) <= median(u) + median(v)."""

    num_vertices = distances.shape[0]
    thresholds = medians[:, jnp.newaxis] + medians[jnp.newaxis, :]

    return (distances <= thresholds) & ~jnp.eye(num_vertices, dtype=jnp.bool_)


def recon_transitions(distances: jax.Array) -> jax.Array:
    """Training-free weight estimate: a global softmax over negative distances."""

    num_vertices = distances.shape[0]
    logits = jnp.where(jnp.eye(num_vertices, dtype=jnp.bool_), -jnp.inf, -distances)

    return jax.nn.softmax(logits.reshape(-1)).reshape(logits.shape)
