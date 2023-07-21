import pandas as pd
import numpy as np
from ..te_tools.calculate_te import calculate_te


def calculate_hte(df, criterion, is_cat, n_bins, t, X_exo, X_end, y, iv, conf_level):
    """
    calculate treatment effects for subsets based on a selection criteria

    Args:
      df: data
      criterion: the column used for selecting subsets
      is_cat: indicate whether the criterion is categorical
      n_bins: split the df equally by n_bins
      t: treatment variable (also the instrument variable)
      X_exo: exogenous covariate variables
      X_end: single endogenous variable
      y: outcome variable
      iv: whether it is iv regression
      conf_level: confidence interval level

    Returns:
      a dataframe with estimates
    """

    results_df = pd.DataFrame()

    if is_cat:
        unique_values = df[criterion].unique()

        for value in unique_values:
            subset = df[df[criterion] == value]
            result_row = pd.DataFrame.from_records(
                [
                    calculate_te(
                        df=subset,
                        t=t,
                        X_exo=X_exo,
                        X_end=X_end,
                        y=y,
                        iv=iv,
                        conf_level=conf_level,
                    )
                ]
            )
            results_df = pd.concat([results_df, result_row], ignore_index=True)

    else:
        quantile_boundaries = np.linspace(0, 1, n_bins + 1)
        quantiles = df[criterion].quantile(quantile_boundaries)

        for i in range(n_bins):
            lower_bound = quantiles.iloc[i]
            upper_bound = quantiles.iloc[i + 1]
            subset = df.loc[
                (df[criterion] >= lower_bound) & (df[criterion] <= upper_bound)
            ]

            if subset.shape[0] == 0:
                print(
                    f"Warning: No data points found in quantile range {lower_bound:.2f} - {upper_bound:.2f}"
                )
            else:
                result_row = pd.DataFrame.from_records(
                    [
                        calculate_te(
                            df=subset,
                            t=t,
                            X_exo=X_exo,
                            X_end=X_end,
                            y=y,
                            iv=iv,
                            conf_level=conf_level,
                        )
                    ]
                )
                results_df = pd.concat([results_df, result_row], ignore_index=True)

    return results_df
