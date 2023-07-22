from typing import Iterator, NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
    
def net_fn(context):
  mlp = hk.Sequential([
      hk.Linear(5), jax.nn.relu,
      hk.Linear(10), jax.nn.relu,
      hk.Linear(5), jax.nn.relu,
      hk.Linear(1),
  ])
  return mlp(context)

class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState

class HK_NN:
    """
    Class for implementation of JAX Haiku MLP

    """
    def __init__(self, network_fn, eta):
        self.network = hk.without_apply_rng(hk.transform(network_fn))
        self.optimiser = optax.adam(eta)

    def cost(self, params, X, labels):
        """Calculating sigmoid cross entropy loss"""
        # apply model
        N = labels.shape[0]
        logits = self.network.apply(params, X)
        losses = optax.sigmoid_binary_cross_entropy(logits, labels)
        # sigmoid binary cross entropy
        # https://optax.readthedocs.io/en/latest/api.html#losses
        #log_p = jax.nn.log_sigmoid(logits)
           # fX = 1 / (1 + np.exp(-X))
           # a = -Y * np.log(fX) - (1 - Y) * np.log(1 - fX)
        #log_not_p = jax.nn.log_sigmoid(-logits)
        return jnp.mean(losses)

    def update(self, state, X, labels):
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(self.cost)(state.params, X, labels)
        updates, opt_state = self.optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(
        params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state)

    
###########################################################################################################################
## hknn = HK_NN(network_fn = net_fn, eta = 0.01)
    
## initial_params = hknn.network.init(jax.random.PRNGKey(seed=0), jnp.asarray([0.  , 1.  , 0.5], dtype=jnp.float32))
## initial_opt_state = hknn.optimiser.init(initial_params)
## state = TrainingState(initial_params, initial_params, initial_opt_state)