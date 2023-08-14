def evaluate_hte(
    df, criterion, p_x, y, x, t, approx_level=None, start_point=None, search_step=None
):
    """

    evaluate the hte model in the following way:
    given a ranking criterion and p_x in (0,100),
    if we achieve p_x % of total lift in x
    by targetting top ranked users,
    it will yield p_y % of total lift in y


    Args:
      df: data
      criterion: the column used for ranking
      p_x: p_x% reduction in x (between 0-100)
      y: the outcome metric
      x: often an endogenous variable instrumented by t
      t: the binary treatment variable
      approx_level: the approximation error level

    Returns:
      a cutoff value for the criterion and
      the contribution to the total lift in y (in percent)
      share of users included
    """
    if approx_level is None:
        # tolerate 0.01 approximation error
        approx_level = 0.01
    if start_point is None:
        # starting from 1% of the data by default
        start_point = int(df.shape[0] * 1 / 100)
    if search_step is None:
        # increase the search_step to decrease searching time
        search_step = max(1, int(df.shape[0] / 10000))

    df = df.sort_values(by=criterion, ascending=False).reset_index(drop=True)

    def calculate_lift(index_cutoff, target):
        df_temp = df.loc[df.index <= index_cutoff]
        n_c = df_temp.groupby(t)[t].count().sort_index(ascending=True)[0]
        target_avg_c, target_avg_t = (
            df_temp.groupby(t)[target].mean().sort_index(ascending=True)
        )
        total_lift = (target_avg_t - target_avg_c) * n_c
        return total_lift

    total_lift_x_all = calculate_lift(df.shape[0], x)
    total_lift_y_all = calculate_lift(df.shape[0], y)

    for i in range(start_point, df.shape[0], search_step):
        if abs(calculate_lift(i, x) / total_lift_x_all - p_x / 100) <= approx_level:
            break

    p_y = 100 * (calculate_lift(i, y) / total_lift_y_all)
    p_users = 100 * i / df.shape[0]
    criterion_cutoff = df.loc[df.index == i, criterion].values[0]

    result_dict = {
        "p_x": p_x,
        "p_y": p_y,
        "p_users": p_users,
        "criterion_cutoff": criterion_cutoff,
        "criterion_name": criterion,
    }

    return result_dict
