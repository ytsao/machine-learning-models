from jax import grad, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def create_dataset():
    X, y = make_regression(n_samples=150, n_features=2, noise=5)
    y = y.reshape((y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], y, c='r')
    plt.scatter(X[:,1], y, c='b')
    plt.show()

    return X_train, X_test, y_train, y_test


def loss_function(w, b, X, y):
    pred = X.dot(w) + b
    return ((pred - y) ** 2).mean()


def main():
    X_train, X_test, y_train, y_test = create_dataset()
    weights = jnp.zeros((X_train.shape[1], 1))
    bias = 0. 
    learning_rate = 0.001
    num_iterations = 3000

    grad_weights = jit(grad(loss_function, argnums=0))
    grad_bias = jit(grad(loss_function, argnums=1))

    for _ in range(num_iterations):
        dW = grad_weights(weights,bias,X_train,y_train)
        db = grad_bias(weights,bias,X_train,y_train)
        # print(loss_function(weights,bias,X_train,y_train))
        weights -= dW*learning_rate
        bias -= db*learning_rate
    

    error = loss_function(weights, bias, X_test, y_test)
    print(f"Error: {error}")

    return 


if __name__ == "__main__":
    main()