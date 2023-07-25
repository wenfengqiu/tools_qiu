import numpy as np
from sklearn.linear_model import LinearRegression

# sklearn does not provide se/cf and is faster


def update_score(
    df,
    subsample_size,
    t,
    X_exo,
    y,
    tau_0,
    minus_tau_0,
    replacement,
):
    sample_index = np.random.choice(
        df.index,
        size=subsample_size,
        replace=replacement,
    )
    sample_df = df.loc[sample_index]
    X = sample_df[[t] + X_exo]
    Y = sample_df[y]
    tau = LinearRegression().fit(X, Y).coef_[0]
    df.loc[sample_index, "score"] += tau - tau_0 * minus_tau_0


def calculate_score(df, t, X_exo, y, **kwargs):
    subsample_size = kwargs.get("subsample_size", 1000)
    replacement = kwargs.get("replacement", False)
    minus_tau_0 = kwargs.get("minus_tau_0", True)
    num_rounds = kwargs.get("num_rounds", 500)
    df["score"] = 0
    X = df[[t] + X_exo]
    Y = df[y]
    tau_0 = LinearRegression().fit(X, Y).coef_[0]
    for i in range(num_rounds):
        update_score(
            df=df,
            subsample_size=subsample_size,
            t=t,
            X_exo=X_exo,
            y=y,
            tau_0=tau_0,
            minus_tau_0=minus_tau_0,
            replacement=replacement,
        )
