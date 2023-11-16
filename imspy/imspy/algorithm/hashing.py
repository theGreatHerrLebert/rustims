import tensorflow as tf
import numpy as np
import warnings


class CosimHasher:
    def __init__(self, target_vector_length, trials: int = 32, len_trial: int = 20, seed: int = 42):

        assert 0 < trials, f'trials variable needs to be greater then 1, was: {trials}'
        assert 0 < len_trial, f'length trial variable needs to be greater then 1, was: {trials}'

        # check
        if 0 < len_trial <= 32:
            self.V = tf.constant(
                np.expand_dims(np.array([np.power(2, i) for i in range(len_trial)]).astype(np.int32), 1))

        elif 32 < len_trial <= 64:
            warnings.warn(f"\nnum bits to hash set to: {len_trial}.\n" +
                          f"using int64 which might slow down computation significantly.")
            self.V = tf.constant(
                np.expand_dims(np.array([np.power(2, i) for i in range(len_trial)]).astype(np.int64), 1))
        else:
            raise ValueError(f"bit number per hash cannot be greater then 64 or smaller 1, was: {len_trial}.")

        self.trials = trials
        self.len_trial = len_trial
        self.seed = seed
        self.target_vector_length = target_vector_length

        np.random.seed(seed)
        size = (len_trial * self.trials, target_vector_length)
        X = np.random.normal(0, 1, size=size).astype(np.float32)
        self.hash_tensor = tf.transpose(tf.constant(X))

    def __repr__(self) -> str:
        return f"CosimHasher(trials={self.trials}, len_trial={self.len_trial}, seed={self.seed}, " \
               f"target_vector_length={self.target_vector_length})"

    def calculate_keys(self, W: tf.Tensor) -> tf.Tensor:

        S = (tf.sign(W @ self.hash_tensor) + 1) / 2

        if self.len_trial <= 32:
            # reshape into window, num_hashes, len_single_hash
            S = tf.cast(tf.reshape(S, shape=(S.shape[0], self.trials, self.len_trial)), tf.int32)

            # calculate int key from binary by base-transform
            H = tf.squeeze(S @ self.V)
            return H
        else:
            # reshape into window, num_hashes, len_single_hash
            S = tf.cast(tf.reshape(S, shape=(S.shape[0], self.trials, self.len_trial)), tf.int64)

            # calculate int key from binary by base-transform
            H = tf.squeeze(S @ self.V)
            return H


class TimsHasher(CosimHasher):
    """
    Class to create hash keys from a given set of weights.

    Args:

        trials (int): number of trials to use for random projection.
        len_trial (int): length of each trial.
        seed (int): seed for random projection.
        resolution (int): resolution of the random projection.
        num_dalton (int): number of dalton to use for random projection.
    """
    def __init__(self, trials: int = 32, len_trial: int = 20, seed: int = 5671, resolution: int = 1, num_dalton: int = 10):
        res_factor = 10 ** resolution
        target_vector_length = num_dalton * res_factor + 1
        super().__init__(target_vector_length, trials, len_trial, seed)
        self.resolution = resolution
        self.num_dalton = num_dalton

    def __repr__(self) -> str:
        return f"TimsHasher(trials={self.trials}, len_trial={self.len_trial}, seed={self.seed}, " \
               f"resolution={self.resolution}, num_dalton={self.num_dalton})"
