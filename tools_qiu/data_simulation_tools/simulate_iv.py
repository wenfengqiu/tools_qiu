import numpy as np
import pandas as pd


def simulate_iv(data_generation_parameters=None, random_seed=10086, log_normal=True):
    """
    simulate IV data.

    Args:
        log_normal (bool): If True, applies exponential transformation to generated samples.

    Returns:
        pd.DataFrame: Simulated IV data.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if data_generation_parameters is None:
        data_generation_parameters = {
            "sample_size": 2000,
            "treatment_proportion": 0.3,
            "normal_mean": np.array([0, 0, 0]),
            # covariance matrix for x1, epsilon, delta
            "normal_cov": np.array([[1, 0.7, -0.5], [0.7, 1, -0.4], [-0.5, -0.4, 1]]),
            # coefficients for x: intercept, treatment_effect, x1, epsilon, delta
            "coefficients": np.array([3, -2, 0.3, 1, 0]),
            # coefficients for y: intercept, x, x1, epsilon, delta
            "iv_coefficients": np.array([400, -3, -0.15, 0, 1]),
        }

    samples = np.random.multivariate_normal(
        data_generation_parameters["normal_mean"],
        data_generation_parameters["normal_cov"],
        size=data_generation_parameters["sample_size"],
    )

    treatment_assignment = np.random.binomial(
        n=1,
        p=data_generation_parameters["treatment_proportion"],
        size=data_generation_parameters["sample_size"],
    )

    df = pd.DataFrame(samples, columns=["x1", "epsilon", "delta"])

    if log_normal:
        df = df.apply(lambda x: np.exp(x))

    df["treatment"] = treatment_assignment
    df["intercept"] = 1
    df["x"] = df[["intercept", "treatment", "x1", "epsilon", "delta"]].dot(
        data_generation_parameters["coefficients"]
    )
    df["y"] = df[["intercept", "x", "x1", "epsilon", "delta"]].dot(
        data_generation_parameters["iv_coefficients"]
    )

    df["obs_index"] = np.arange(1, data_generation_parameters["sample_size"] + 1)

    return df
