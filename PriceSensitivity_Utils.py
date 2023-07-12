#The PriceSensitivity contains methods for computing the price sensitivity curve and simulating a dataset.

#The price_sensitivity_curve method computes the price sensitivity curve. It takes an input value r, slope parameter a, and horizontal shift parameter b. The input value is scaled to the (0, 1) space, and the result is computed using the sigmoid function.

#The simulate_dataset method simulates a dataset. It takes the number of samples N, an array of features features, a list of price values price_list, a list of price sensitivity parameters price_sensitivity_parms, and a random seed seed. The method performs random sampling of features, price values, and client groups. It computes the price sensitivity for each client group using the price sensitivity curve and the corresponding parameters. Finally, it assembles the simulated dataset by stacking the feature components, price values, and sale values.

#Please note that the code assumes the necessary libraries (jnp, random, jax.random) have been imported or are available in the environment.

# Numpy API with hardware acceleration and automatic differentiation
from jax import numpy as jnp
import jax

def price_sensitivity_curve(r, a, b):
    """
    Computes the price sensitivity curve.

    Args:
    - r (float): Input value.
    - a (float): Slope parameter.
    - b (float): Horizontal shift parameter.

    Returns:
    - float: Result of the price sensitivity curve computation.
    """
    # Scaling to (0,1) space
    r = a * 20 * (r - 0.5 - b)
    return 1 - (1 / (1 + jnp.exp(-r)))

def simulate_dataset(N, features, price_list, price_sensitivity_parms, seed):
    """
    Simulates a dataset.

    Args:
    - N (int): Number of samples.
    - features (ndarray): Array of features.
    - price_list (list): List of price values.
    - price_sensitivity_parms (list): List of price sensitivity parameters.
    - seed (int): Random seed.

    Returns:
    - ndarray: Simulated dataset.
    """
 
    rng_f, rng_p, rng_s = random.split(jax.random.PRNGKey(seed), num=3)

    random_indices_feature = jax.random.randint(rng_f, shape=(N,), minval=0, maxval=features.shape[0])
    array_feature = jnp.array([features[i] for i in random_indices_feature])
    array_feature1 = jnp.array([i[0] for i in array_feature])
    array_feature2 = jnp.array([i[1] for i in array_feature])

    random_indices_price = jax.random.randint(rng_p, shape=(N,), minval=0, maxval=len(price_list))
    array_price = jnp.array([price_list[i] for i in random_indices_price])

    array_client_group = [0 if jnp.all(f == jnp.asarray([0, 1])) else 1 for f in array_feature]

    array_p = jnp.asarray([price_sensitivity_curve(array_price[i], price_sensitivity_parms[array_client_group[i]][0], price_sensitivity_parms[array_client_group[i]][1]) for i in range(len(array_client_group))])
    array_sale = jnp.asarray(jax.random.bernoulli(rng_s, p=array_p, shape=None))
    
    ds = jnp.stack([array_feature1, array_feature2, array_price, array_sale], axis=0)
    
    return ds