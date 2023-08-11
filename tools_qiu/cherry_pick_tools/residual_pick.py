import numpy as np
from sklearn.linear_model import LinearRegression

# cherry pick based on residuals


def residual_pick(df, t, X_exo, y, pick_share, include_t=True):
    """
    pick the pick_share % of the data
    (with ties, obtain more data)

    Args:
      df: data
      pick_share: between 0-100
      include_t: optional when the ATE = 0

    Returns:
      df with the residuals
      and the addtional column indicating whether a row is picked
    """
    df = df.copy()
    if X_exo is None:
        Y = df[y]
        if include_t:
            X = df[[t]]
            model = LinearRegression().fit(X, Y)
            res = Y - model.predict(X)
        else:
            res = Y - Y.mean()
    else:
        if include_t:
            X = df[[t] + X_exo]
        else:
            X = df[X_exo]
        Y = df[y]
        model = LinearRegression().fit(X, Y)
        res = Y - model.predict(X)
    df = df.assign(residual=res)

    threshold_0 = df.loc[df[t] == 0, "residual"].quantile(pick_share / 100)
    threshold_1 = df.loc[df[t] == 1, "residual"].quantile(1 - pick_share / 100)

    df["residual_pick"] = 0
    df["residual_pick"] = df.apply(
        lambda row: 1
        if (row[t] == 1 and row["residual"] >= threshold_1)
        or (row[t] == 0 and row["residual"] <= threshold_0)
        else 0,
        axis=1,
    )

    return df
