import numpy as np
import pandas as pd
from ..te_tools.calculate_te import calculate_te


def simulate_iv_base_rate_neglect():
    """
    simulate a df for demonstrating the base rate neglect in iv estimation

    Args:

    Returns:
      a list [dataframe, results_in_dictionary]
    """

    # three types of users
    # each have 1000 users
    # type 1: dy/dx = 1/2
    # type 2: dy/dx = 1/3
    # type 3: dy/dx = 101/300

    # data generating process
    # x, epsilon, delta
    samples = np.random.multivariate_normal(
        np.array([0, 0, 0]),
        np.array([[1, 0.7, -0.5], [0.7, 1, -0.4], [-0.5, -0.4, 1]]),
        size=30000,
    )
    df = pd.DataFrame(samples, columns=["x1", "epsilon", "delta"])
    # add some exogenous variable
    df["x2"] = np.random.standard_normal(size=30000)
    # assign types
    df["type"] = 0
    df.loc[:9999, "type"] = 1
    df.loc[10000:19999, "type"] = 2
    df.loc[20000:, "type"] = 3
    # assign treatment
    df["treatment"] = np.random.binomial(
        n=1,
        p=0.5,
        size=30000,
    )
    df.loc[df["type"] == 3, ["x1", "x2", "epsilon", "delta"]] *= 100

    # dx/dt
    def calculate_x(row):
        if row["type"] == 1:
            return row["x1"] + row["treatment"] * 2 + row["epsilon"]
        elif row["type"] == 2:
            return row["x1"] + row["treatment"] * 3 + row["epsilon"]
        else:
            return row["x1"] + row["treatment"] * 300 + row["epsilon"]

    df["x"] = df.apply(calculate_x, axis=1)

    # dx/dy
    def calculate_y(row):
        if row["type"] == 1:
            return (1 / 2) * row["x"] + row["x2"] + row["delta"]
        elif row["type"] == 2:
            return (1 / 3) * row["x"] + row["x2"] + row["delta"]
        else:
            return (101 / 300) * row["x"] + row["x2"] + row["delta"]

    df["y"] = df.apply(calculate_y, axis=1)

    # run iv seperately
    te1 = calculate_te(
        df.loc[df["type"] == 1], "treatment", ["x2"], "x", "y", iv=True, conf_level=0.95
    )["te"]

    te2 = calculate_te(
        df.loc[df["type"] == 2], "treatment", ["x2"], "x", "y", iv=True, conf_level=0.95
    )["te"]

    te3 = calculate_te(
        df.loc[df["type"] == 3], "treatment", ["x2"], "x", "y", iv=True, conf_level=0.95
    )["te"]

    # run iv for 1&2
    te1_2 = calculate_te(
        df.loc[df["type"] < 3], "treatment", ["x2"], "x", "y", iv=True, conf_level=0.95
    )["te"]

    # run iv for 1&3
    te1_3 = calculate_te(
        df.loc[df["type"] != 2], "treatment", ["x2"], "x", "y", iv=True, conf_level=0.95
    )["te"]

    result_dict = {
        "iv_effect_type_1": te1,
        "iv_effect_type_2": te2,
        "iv_effect_type_3": te3,
        "iv_effect_type_1_plus_2": te1_2,
        "iv_effect_type_1_plus_3": te1_3,
    }

    return [df, result_dict]
