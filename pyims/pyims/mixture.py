import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


class GaussianMixtureModel(tf.Module):

    def __init__(self, num_components: int, data_dim: int, prior_stdevs=None, data=None,
                 init_means=None, init_stds=None):
        """
        Initialize the Gaussian Mixture Model.

        Parameters:
        - num_components: Number of Gaussian components.
        - data_dim: Dimensionality of the data.
        - prior_stdevs (optional): Prior knowledge about cluster extensions (standard deviations).
        - lambda_scale: Regularization strength for the scales.
        - data (optional): If provided and no init_means is given, initialize the component means by randomly selecting from this data.
        - init_means (optional): Explicit initial means for the components.
        - init_scales (optional): Explicit initial scales (variances) for the components.
        """

        # Initialize the locations of the GMM components
        super().__init__()

        if init_means is not None:
            assert init_means.shape == (num_components,
                                        data_dim), f"init_means should have shape [num_components, data_dim], but got {init_means.shape}"
            init_locs = tf.convert_to_tensor(init_means, dtype=tf.float32)

        elif data is not None:
            indices = np.random.choice(data.shape[0], size=num_components, replace=True,)
            init_locs = tf.convert_to_tensor(data[indices], dtype=tf.float32)

        else:
            init_locs = tf.random.normal([num_components, data_dim])

        self.locs = tf.Variable(init_locs, name="locs")

        if init_stds is not None:
            init_stds_vals = tf.repeat(init_stds, num_components, axis=0)
        else:
            init_stds_default = [[3, 0.01, 0.01]]
            init_stds_vals = tf.repeat(init_stds_default, num_components, axis=0)

        self.scales = tf.Variable(tf.math.log(init_stds_vals), name="scales")

        # Initialize the weights of the GMM components
        self.weights = tf.Variable(tf.ones([num_components]), name="weights")

        # Set the prior scales and regularization strength
        if prior_stdevs is not None:
            init_prior_stds = tf.repeat(prior_stdevs, num_components, axis=0)
            self.prior_stdevs = init_prior_stds

    def __call__(self, data):
        """
        Calculate the log likelihood of the data given the current state of the GMM.
        """

        # Constructing the multivariate normal distribution with diagonal covariance
        components_distribution = tfd.Independent(tfd.Normal(loc=self.locs, scale=tf.math.exp(self.scales)),
                                                  reinterpreted_batch_ndims=1)

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=tf.nn.log_softmax(self.weights)),
            components_distribution=components_distribution)

        return gmm.log_prob(data)

    def fit(self, data, weights=None, num_steps=200, learning_rate=0.05, lambda_scale=0.01, verbose=True):
        """
        Fit the Gaussian Mixture Model to the data.

        Parameters:
        - data: Input data of shape [n_samples, n_features].
        - weights (optional): Weights for each sample.
        - num_steps: Number of optimization steps.
        - learning_rate: Learning rate for the optimizer.
        """
        if weights is None:
            weights = tf.ones(len(data))

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                log_likelihood = self.__call__(data)
                loss = -tf.reduce_sum(log_likelihood * weights)  # Weighted negative log likelihood

                # Add regularization based on prior scales if provided

                if self.prior_stdevs is not None:
                    scale_diff = tf.math.exp(self.scales) - self.prior_stdevs
                    reg_loss = lambda_scale * tf.reduce_sum(scale_diff * scale_diff)
                    loss += reg_loss

            gradients = tape.gradient(loss, [self.locs, self.scales, self.weights])
            optimizer.apply_gradients(zip(gradients, [self.locs, self.scales, self.weights]))
            return loss

        for step in range(num_steps):
            loss = train_step()
            if step % 100 == 0 and verbose:
                tf.print("step:", step, "log-loss:", loss)

    def __mixture(self):
        """
        Creates a Gaussian Mixture Model from the current parameters (weights, means, and covariances).
        """
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=tf.nn.log_softmax(self.weights)),
            components_distribution=tfd.Independent(tfd.Normal(loc=self.locs, scale=tf.math.exp(self.scales)),
                                                    reinterpreted_batch_ndims=1))

    @property
    def variances(self):
        """Returns the actual variances (squared scales) of the Gaussian components."""
        return tf.math.exp(2 * self.scales)

    @property
    def stddevs(self):
        """Returns the actual standard deviations (scales) of the Gaussian components."""
        return tf.math.exp(self.scales)

    def predict_proba(self, data):
        gmm = self.__mixture()

        # Calculate the log probabilities for each data point for each component
        log_probs = gmm.components_distribution.log_prob(tf.transpose(data[..., tf.newaxis], [0, 2, 1]))

        # Convert log probabilities to unnormalized probabilities
        unnormalized_probs = tf.exp(log_probs)

        # Normalize the probabilities
        probs_sum = tf.reduce_sum(unnormalized_probs, axis=-1, keepdims=True)
        normalized_probs = unnormalized_probs / probs_sum

        return normalized_probs.numpy()

    def predict(self, data):
        """Get the cluster ids under the current mixture model"""
        return np.argmax(self.predict_proba(data), axis=1)

    def sample(self, n_samples=1):
        gmm = self.__mixture()

        # Sample from the Gaussian Mixture Model
        samples = gmm.sample(n_samples)

        return samples
