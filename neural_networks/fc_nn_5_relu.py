import time

import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.scipy.special import logsumexp

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    print(keys)
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def relu(x):
    return jnp.maximum(0, x)


def predict(params, input):
    activatation = input
    for w, b in params[:-1]:
        activatation = relu(jnp.dot(w, activatation) + b)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activatation) + final_b
    return logits - logsumexp(logits)


def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, data, targets):
    target_class = jnp.argmax(targets, axis=1)
    batched_predict = vmap(predict, in_axes=(None, 0))
    predicted_class = jnp.argmax(batched_predict(params, data), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, data, targets):
    preds = predict(params, data)
    return -jnp.mean(preds * targets)


@jit
def update(params, x, y, step_size):
    grads = grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def get_train_batches(data_dir, batch_size):
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name="mnist", split="train", as_supervised=True, data_dir=data_dir)
    # You can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batch_size).prefetch(1)
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)


def main():
    layer_sizes: list[int] = [784, 512, 512, 10]
    step_size: float = 0.01
    num_epochs: int = 10
    batch_size: int = 128
    n_targets: int = 10
    params = init_network_params(layer_sizes, random.key(0))

    data_dir: str = "/tmp/tfds"
    mnist_data, info = tfds.load(
        name="mnist", batch_size=-1, data_dir=data_dir, with_info=True
    )
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data["train"], mnist_data["test"]
    num_labels = info.features["label"].num_classes
    h, w, c = info.features["image"].shape
    num_pixels = h * w * c

    # full train set
    train_images, train_labels = train_data["image"], train_data["label"]
    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot(train_labels, num_labels)

    # full test set
    test_images, test_labels = test_data["image"], test_data["label"]
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot(test_labels, num_labels)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in get_train_batches(data_dir, batch_size):
            x = jnp.reshape(x, (len(x), num_pixels))
            y = one_hot(y, num_labels)

            params = update(params, x, y, step_size)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)


if __name__ == "__main__":
    main()
