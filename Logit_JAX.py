# The logistic method applies the logistic function to map input values to the range [0, 1].

# The predict_proba method predicts the probabilities using logistic regression. It takes the intercept term c, coefficient vector w, and input features X. It uses the logistic function to compute the probabilities.

# The cost method computes the cost function for logistic regression. It takes the intercept term c, coefficient vector w, input features X, true labels y, and optional parameters eps and lmbd. It calculates the cost by summing the element-wise product of the true labels and the logarithm of the predicted probabilities, along with the element-wise product of (1 - y) and the logarithm of (1 - predicted probabilities). It also includes a regularization term.

# The params_init method initializes the initial action values for the logistic regression model. It takes the number of features n_feat and returns a dictionary containing the initial values for c, w, and cost.

# The logit_update method performs online learning, which is stochastic gradient descent (SGD) with reaction and cost function. It takes the current parameters params, input features X, and true labels sold. It updates the intercept term c and coefficient vector w using gradient descent based on the cost function. It returns a dictionary containing the updated parameters c, w, and the current cost.

# Please note that the code assumes the necessary libraries (jnp) and the grad function have been imported or are available in the environment.

# Numpy API with hardware acceleration and automatic differentiation
from jax import numpy as jnp
# Creates a function that evaluates the gradient of fun.
from jax import grad

def logistic(r):
    """
    Logistic function to map input values to the range [0, 1].
    """
    return 1 / (1 + jnp.exp(-r))

def predict_proba(c, w, X):
    """
    Predicts the probability using logistic regression.

    Args:
    - c (float): Intercept term.
    - w (ndarray): Coefficient vector.
    - X (ndarray): Input features.

    Returns:
    - ndarray: Predicted probabilities.
    """
    return logistic(jnp.dot(X, w) + c)

def cost(c, w, X, y, eps=1e-14, lmbd=0.1):
    """
    Computes the cost function for logistic regression.

    Args:
    - c (float): Intercept term.
    - w (ndarray): Coefficient vector.
    - X (ndarray): Input features.
    - y (ndarray): True labels.
    - eps (float, optional): Small value to avoid logarithmic errors. Defaults to 1e-14.
    - lmbd (float, optional): Regularization parameter. Defaults to 0.1.

    Returns:
    - float: Cost value.
    """
    n = y.size
    p = predict_proba(c, w, X)
    p = jnp.clip(p, eps, 1 - eps)  # Bound the probabilities within (0, 1) to avoid log(0)

    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) / n + 0.5 * lmbd * (
        jnp.dot(w, w) + c * c
    )

def params_init(n_feat):
    """
    Returns the initial action values.
    """
    return {
        'c': 1.0,
        'w': 1.0e-5 * jnp.ones(n_feat + 1),
        'cost': 0
    }

def logit_update(params, X, sold, eta=0.01):
    """
    Performs online-learning, i.e. SGD with reaction and cost function.
    """
    c_current = params['c']
    w_current = params['w']
    cost_current = cost(c_current, w_current, X, sold)

    c_update = c_current - eta * grad(cost, argnums=0)(c_current, w_current, X, sold)
    w_update = w_current - eta * grad(cost, argnums=1)(c_current, w_current, X, sold)

    return {
        # Perform Gradient Descent on c
        'c': c_update,

        # Perform Gradient Descent on w
        'w': w_update,

        'cost': cost_current
    }