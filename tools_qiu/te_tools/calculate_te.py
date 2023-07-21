from linearmodels.iv import IV2SLS
import statsmodels.api as sm


def calculate_te(df, t, X_exo, X_end, y, iv, conf_level):
    """
    calculate treatment effects

    Args:
      df: data
      t: treatment variable (also the instrument variable)
      X_exo: exogenous covariate variables
      X_end: single endogenous variable
      y: outcome variable
      iv: whether it is iv regression
      conf_level: confidence interval level

    Returns:
      a dictionary with estimates
    """
    y_c, y_t = df.groupby(t)[y].mean().sort_values(ascending=True)
    end_c, end_t = 0, 0
    dependent = df[y]
    if iv:
        end_c, end_t = df.groupby(t)[X_end].mean().sort_values(ascending=True)
        exo = df[X_exo]
        exo = sm.add_constant(exo)
        end = df[X_end]
        instruments = df[t]
        model = IV2SLS(dependent, exo, end, instruments)
        result = model.fit()
        te = result.params[X_end]
        te_low, te_high = result.conf_int(level=conf_level).loc[X_end]
        std = result.std_errors[X_end]
    else:
        exo = df[[t] + X_exo]
        exo = sm.add_constant(exo)
        model = IV2SLS(dependent, exo, None, None)
        result = model.fit()
        te = result.params[t]
        te_low, te_high = result.conf_int(level=conf_level).loc[t]
        std = result.std_errors[t]

    result_dict = {
        "data_size": df.shape[0],
        "y": y,
        "y_c": y_c,
        "y_t": y_t,
        "end": X_end,
        "end_c": end_c,
        "end_t": end_t,
        "te": te,
        "std": std,
        "te_low": te_low,
        "te_high": te_high,
        "iv": iv,
    }

    return result_dict
