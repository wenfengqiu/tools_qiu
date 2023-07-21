def pick_top(df, criterion, top_p, subset=False):
    """
    pick the top_p share of the data
    (with ties, obtain less data)
    (Warning: this function mutate the df inplace)

    Args:
      df: data
      criterion: the column used for ranking
      top_p: between 0-100
      subset: whether return a subset and keep the original df

    Returns:
      df with the addtional column indicating top or not
      or a subset with the top data
    """

    threshold = df[criterion].quantile(1 - top_p / 100)
    col_name = "top_" + criterion
    if subset:
        df_subset = df.loc[df[criterion] > threshold]
        return df_subset
    else:
        df[col_name] = df[criterion].apply(
            lambda x: "top {}%".format(top_p) if x > threshold else "rest"
        )
