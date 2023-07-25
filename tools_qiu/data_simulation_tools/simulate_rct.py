import numpy as np
import pandas as pd


def simulate_rct(data_generation_parameters=None, random_seed=10086, log_normal=True):
    """
    simulate RCT data.

    Args:
        log_normal (bool): If True, applies exponential transformation to generated samples.

    Returns:
        pd.DataFrame: Simulated RCT data.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if data_generation_parameters is None:
        data_generation_parameters = {
            "sample_size": 2000,
            "treatment_proportion": 0.3,
            # coefficients ordering: intercept, treatment_effect, x1, x2, epsilon
            "coefficients": np.array([0.1, 0.2, 0.3, -0.2, 1]),
            "normal_mean": np.array([0, 0, 0]),
            "normal_cov": np.array([[1, 0.7, -0.5], [0.7, 1, -0.4], [-0.5, -0.4, 1]]),
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

    df = pd.DataFrame(samples, columns=["x1", "x2", "epsilon"])

    if log_normal:
        df = df.apply(lambda x: np.exp(x))

    df["treatment"] = treatment_assignment
    df["intercept"] = 1
    df["y"] = df[["intercept", "treatment", "x1", "x2", "epsilon"]].dot(
        data_generation_parameters["coefficients"]
    )

    df["obs_index"] = np.arange(1, data_generation_parameters["sample_size"] + 1)

    return df
