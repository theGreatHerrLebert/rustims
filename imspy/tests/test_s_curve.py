import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import binom
from tqdm import tqdm

from imspy.algorithm.hashing import TimsHasher

np.random.seed(0)


def get_signal_noise(sigma, n_windows, n_bins):
    # generate F and F_prime
    F = np.random.randn(n_windows, n_bins)

    noise = np.random.randn(1, n_bins) * sigma
    F_prime = F + noise

    signal = np.zeros((n_windows, n_bins))
    noise = np.zeros_like(signal)

    for i in np.arange(n_windows):
        mat_i = get_random_rotation(n_bins)

        new_sig = mat_i @ F[0, :]
        signal[i, :] = new_sig

        new_noise = mat_i @ F_prime[0, :]
        noise[i, :] = new_noise

    return signal, noise


def get_p_estimate(ors, ands, F, F_prime, seed):
    hasher = TimsHasher(trials=ors, len_trial=ands, seed=seed, num_dalton=10, resolution=1)
    H = hasher.calculate_keys(F)
    H_p = hasher.calculate_keys(F_prime)
    return H[np.any((H == H_p).numpy(), axis=1)].shape[0] / H.shape[0], H, H_p


def and_or(p, n, m):
    # n times AND
    # m times OR
    return 1.0 - np.power((1.0 - np.power(p, n)), m)


def get_p_model(s, ands, ors):
    pSim = 1.0 - (np.arccos(s)) / np.pi
    return and_or(pSim, ands, ors)


def get_random_rotation(dim):
    m = np.random.randn(dim, dim)
    m_s = (m + m.T) / np.sqrt(2)
    v, mat = np.linalg.eig(m_s)
    return mat


def get_estimates(ors, ands, signal, noise, seed, n_windows):
    p_est, H, H_p = get_p_estimate(ors, ands, signal, noise, seed)

    sims = []
    for (s, n) in zip(signal, noise):
        sim = cosine_similarity(s.reshape(1, -1), n.reshape(1, -1))[0][0]
        sims.append(sim)

    sim_median = np.median(sims)

    p_mod = get_p_model(sim_median, ands, ors)
    inter = binom.ppf([0.025, 0.975], n_windows, p_mod) / n_windows

    return sim_median, p_est, p_mod, inter


def test_s_curve():
    """
    """
    # Arrange
    n_windows = 3000
    n_bins = 101

    ors = 128
    ands = 31

    seed = 1354656

    # Action
    dict_values = {}

    for sigma in tqdm([0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                      ncols=100,
                      desc="iterating collision probabilities"):
        signal, noise = get_signal_noise(sigma, n_windows, n_bins)
        sim_median, p_est, p_mod, inter = get_estimates(ors, ands, signal, noise, seed, n_windows)
        dict_values[sigma] = (sim_median, p_est, p_mod, inter)

    # Assert
    for (k, (sim_median, p_est, p_mod, inter)) in dict_values.items():
        assert inter[0] <= p_est <= inter[1]


def main():
    test_s_curve()


if __name__ == '__main__':
    main()
