import equinox as eqx
import jax
import jax.numpy as jnp

import embedex.distributions.flows.flowee as flowee
from embedex.distributions.conditioning import (
    CrossAttentionBlock,
    batch_layer_norm,
    batch_linear,
)
from embedex.distributions.flows.flow_matching import sinusoidal_embedding


class DiscreteDenoiserNet(eqx.Module):
    """Time-conditioned denoiser network for discrete flow matching.

    Takes integer-valued noisy tokens `x_t` and a scalar time `t`,
    one-hot encodes them internally, and predicts logits for the clean
    data distribution `p(x_1 | x_t, t)` independently for each variable.
    """

    time_embed_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    net: eqx.Module

    def __call__(
        self,
        x_t: jax.Array,
        t: jax.Array,
        sideinfo: jax.Array | None = None,
    ) -> jax.Array:
        """Predict logits for clean data given noisy tokens.

        Args:
            x_t: Integer class indices, shape `(num_variables,)`.
            t: Scalar time in `[0, 1]`.
            sideinfo: Optional conditioning vector.

        Returns:
            Logits of shape `(num_variables, num_classes)`.
        """

        x_t_onehot = jax.nn.one_hot(x_t, self.num_classes).reshape(-1)
        t_embed = sinusoidal_embedding(t, self.time_embed_dim)
        net_input = jnp.concatenate([x_t_onehot, t_embed])

        raw = self.net(net_input, sideinfo) if sideinfo is not None else self.net(net_input)
        return raw.reshape(self.num_variables, self.num_classes)


class AttentionDenoiserNet(eqx.Module):
    """Cross-attention denoiser for discrete flow matching.

    Each *variable* in the categorical sample space is a token (sequence
    length = `num_variables`).

    Token representations are composed additively from:

    - A learnable variable (positional) embedding
    - The class embedding of the current noisy value for that variable
    - A time embedding (broadcast to all tokens)
    - A **global conditioning projection** (broadcast to all tokens)

    Conditioning is injected through two complementary paths:

    1. **Global**: the full conditioning vector is projected and added to
       every variable token, giving immediate access to the complete signal
       (analogous to the broadcast time embedding).
    2. **Structured**: the conditioning vector is split into
       `num_cond_tokens` chunks, each projected into the token space and
       augmented with a learned positional embedding.  These serve as
       key/value targets for cross-attention at every layer, allowing
       richer, token-level conditioning.

    Each variable token independently predicts logits over its
    `num_classes` possible values via a shared output projection.
    """

    time_embed_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    num_cond_tokens: int = eqx.field(static=True)
    cond_token_raw_dim: int = eqx.field(static=True)
    rel_node_dim: int = eqx.field(static=True)
    moment_dim: int = eqx.field(static=True)

    class_embeddings: jax.Array
    variable_embeddings: jax.Array
    time_proj: eqx.nn.Linear
    cond_token_proj: eqx.nn.Linear
    cond_time_proj: eqx.nn.Linear
    cond_global_proj: eqx.nn.Linear
    cond_pos_embeddings: jax.Array
    blocks: tuple[CrossAttentionBlock, ...]
    final_norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear

    def __call__(
        self,
        x_t: jax.Array,
        t: jax.Array,
        sideinfo: jax.Array | None = None,
    ) -> jax.Array:
        """Predict logits for clean data given noisy tokens.

        Args:
            x_t: Integer class indices, shape `(num_variables,)`.
            t: Scalar time in `[0, 1]`.
            sideinfo: Optional conditioning vector, shape
                `(num_cond_tokens * cond_token_raw_dim,)`.

        Returns:
            Logits of shape `(num_variables, num_classes)`.
        """

        class_signal = self.class_embeddings[x_t]

        t_embed = sinusoidal_embedding(t, self.time_embed_dim)
        t_signal = self.time_proj(t_embed)
        tokens = self.variable_embeddings + class_signal + t_signal[None, :]

        if sideinfo is not None:
            sideinfo_flat = jnp.ravel(sideinfo)
            cond_t = self.cond_time_proj(t_embed)
            if self.rel_node_dim > 0:
                n = self.rel_node_dim
                mu = sideinfo_flat[: self.moment_dim]
                jrows = sideinfo_flat[self.moment_dim :].reshape(n, n)
                tokens = tokens + self.cond_global_proj(mu)[None, :]
                pernode = jnp.concatenate(
                    [jnp.broadcast_to(mu, (n, self.moment_dim)), jrows], axis=-1
                )
                node_signal = batch_linear(pernode, self.cond_token_proj)
                tokens = tokens + node_signal
                cond_tokens = node_signal + cond_t[None, :] + self.cond_pos_embeddings
            else:
                cond_global = self.cond_global_proj(sideinfo_flat)
                tokens = tokens + cond_global[None, :]
                cond_raw = sideinfo_flat.reshape(self.num_cond_tokens, self.cond_token_raw_dim)
                cond_tokens = batch_linear(cond_raw, self.cond_token_proj)
                cond_tokens = cond_tokens + cond_t[None, :] + self.cond_pos_embeddings
        else:
            cond_tokens = jnp.zeros((1, self.token_dim))

        for block in self.blocks:
            tokens = block(tokens, cond_tokens)

        tokens = batch_layer_norm(tokens, self.final_norm)
        logits = batch_linear(tokens, self.output_proj)

        return logits


class DiscreteFlowMatchingModel(flowee.Flow):
    """Discrete Flow Matching generative model for categorical data.

    Implements the Discrete Flow Matching framework (Gat et al., 2024,
    *Flow Matching for Discrete Data*) for learning generative models over
    discrete / categorical data, conditioned on an embedding vector.

    Each data point `x` consists of `num_variables` integer-valued
    categorical variables, each taking values in `{0, ..., num_classes-1}`.

    **Forward (noising) process**: at time `t ∈ [0, 1]` each variable
    independently retains its clean value `x_1` with probability `t`,
    or is replaced by a uniform random token with probability `1 - t`.

    The network learns a *denoiser*: given noisy tokens `x_t` and time
    `t`, predict the clean-data distribution `p_θ(x_1 | x_t, t)`.

    **Training loss**: cross-entropy between the predicted distribution
    and the true `x_1`.

    **Sampling**: Euler discretisation of the discrete probability flow,
    where each step mixes "stay" with the denoiser prediction.
    """

    denoiser_net: eqx.Module
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)

    def _to_int(self, x: jax.Array) -> jax.Array:
        return jnp.clip(jnp.round(x.astype(jnp.float32)).astype(jnp.int32), 0, self.num_classes - 1)

    def _corrupt(self, x_int: jax.Array, t: jax.Array, key: jax.Array) -> jax.Array:
        """Sample `x_t` from the forward process `q(x_t | x_1)`.

        Each variable independently keeps its clean value with probability
        `t` or is replaced with a uniform random token with probability
        `1 - t`.
        """

        mask_key, noise_key = jax.random.split(key, 2)
        keep = jax.random.uniform(mask_key, (self.num_variables,)) < t
        noise = jax.random.randint(
            noise_key,
            (self.num_variables,),
            minval=0,
            maxval=self.num_classes,
        )
        return jnp.where(keep, x_int, noise)

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Corrupt data toward noise (full corruption at `t = 0`)."""

        x_int = self._to_int(x)
        if key is not None:
            x_noisy = self._corrupt(x_int, jnp.array(0.0), key)
        else:
            x_noisy = jnp.zeros(self.num_variables, dtype=jnp.int32)
        return x_noisy.astype(jnp.float32), ldj

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Generate from noise via deterministic (argmax) Euler steps."""

        dt = 1.0 / self.num_steps
        ts = jnp.linspace(0.0, 1.0 - dt, self.num_steps)
        x_t = jnp.zeros(self.num_variables, dtype=jnp.int32)

        def step(x_t, t):
            logits = self.denoiser_net(x_t, t, sideinfo)
            probs = jax.nn.softmax(logits, axis=-1)

            rate = jnp.minimum(dt / jnp.maximum(1.0 - t, 1e-4), 1.0)
            x_t_oh_var = jax.nn.one_hot(x_t, self.num_classes)
            trans_probs = (1.0 - rate) * x_t_oh_var + rate * probs

            x_new = jnp.argmax(trans_probs, axis=-1).astype(jnp.int32)
            return x_new, None

        x_final, _ = jax.lax.scan(step, x_t, ts)
        return x_final.astype(jnp.float32), ldj

    def loss(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Discrete flow matching loss (cross-entropy).

        Samples `t ~ U(ε, 1−ε)`, corrupts `x_1` to `x_t` via the
        forward process, and trains the denoiser to recover `x_1`.
        """

        assert key is not None
        t_key, noise_key = jax.random.split(key, 2)

        eps = 1e-4
        t = jax.random.uniform(t_key, minval=eps, maxval=1.0 - eps)

        x_int = self._to_int(x)
        x_t = self._corrupt(x_int, t, noise_key)

        logits = self.denoiser_net(x_t, t, sideinfo)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(log_probs[jnp.arange(self.num_variables), x_int])

        return ce, 0.0, 0.0

    def sample(
        self,
        sideinfo: jax.Array | None,
        params: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """Generate a sample via Euler discretisation of the discrete flow.

        Starting from uniform noise at `t = 0`, iteratively apply
        the learned transition until `t ≈ 1`.
        """

        dt = 1.0 / self.num_steps
        ts = jnp.linspace(0.0, 1.0 - dt, self.num_steps)

        init_key, scan_key = jax.random.split(key, 2)
        x_t = jax.random.randint(
            init_key,
            (self.num_variables,),
            minval=0,
            maxval=self.num_classes,
        )

        step_keys = jax.random.split(scan_key, self.num_steps)

        def step(x_t, t_and_key):
            t, step_key = t_and_key

            logits = self.denoiser_net(x_t, t, sideinfo)
            probs = jax.nn.softmax(logits, axis=-1)

            rate = jnp.minimum(dt / jnp.maximum(1.0 - t, 1e-4), 1.0)
            x_t_oh_var = jax.nn.one_hot(x_t, self.num_classes)
            trans_probs = (1.0 - rate) * x_t_oh_var + rate * probs

            keys = jax.random.split(step_key, self.num_variables)
            x_new = jax.vmap(
                lambda k, p: jax.random.categorical(k, jnp.log(jnp.maximum(p, 1e-30)))
            )(keys, trans_probs)

            return x_new, None

        x_final, _ = jax.lax.scan(step, x_t, (ts, step_keys))
        return x_final.astype(jnp.float32)

    def log_prob(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Not available: discrete flow matching has no tractable probability for a state.

        Reading the denoiser on the clean state at `t` close to 1 is no substitute, because a
        good denoiser returns that state with probability close to 1 whatever the state. Compare
        generated samples instead.
        """

        raise NotImplementedError(
            'Discrete flow matching has no tractable log-probability; compare generated samples'
        )
