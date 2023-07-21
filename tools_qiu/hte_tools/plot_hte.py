import matplotlib.pyplot as plt


def plot_hte(
    df,
    fig_size=(9, 6),
    dpi=200,
    show_size=True,
    percent=False,
    criterion=" ",
    note=" ",
    save_path=None,
):
    """
    plot the hte for all subsets

    Args:
      df: data from calculate_hte
      criterion: the column used for selecting subsets
      show_size: show the data size for each subset
      note: optional note in the bottom of the plot

    Returns:
      None
    """
    subset_name = df["subset_name"]
    data_size = df["data_size"]
    te = df["te"]
    te_low = df["te_low"]
    te_high = df["te_high"]

    if percent:
        te = 100 * te / df["y_c"]
        te_low = 100 * te_low / df["y_c"]
        te_high = 100 * te_high / df["y_c"]

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # plot the error bars
    ax.errorbar(
        range(len(subset_name)),
        te,
        yerr=[te - te_low, te_high - te],
        fmt="o",
        capsize=5,
        markersize=5,
        elinewidth=1,
        color="C0",
        ecolor="black",
    )

    ax.set_xticks(range(len(subset_name)))
    ax.set_xticklabels(subset_name, rotation=45, ha="right")

    if show_size:
        # plot the data_size text close to each error bar
        for i, size in enumerate(data_size):
            ax.text(
                i,
                te[i] + (te_high[i] - te_low[i]) * 0.05,
                f"n={round(size/1000,2)}k",
                ha="center",
                va="bottom",
            )

    # add labels and title
    ax.set_xlabel("Subset Name")
    if percent:
        ax.set_ylabel("Treatment Effect in %")
    else:
        ax.set_ylabel("Treatment Effect")

    if df["iv"][0]:
        ax.set_title(
            "Treatment Effect of {} on {} by Selected Subsets with Criterion ".format(
                df["end"][0], df["y"][0]
            )
            + criterion
        )
    else:
        ax.set_title(
            "Treatment Effect on {} by Selected Subsets with Criterion ".format(
                df["y"][0]
            )
            + criterion
        )

    # add optional note for convenience
    ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center")

    # save the plot if needed
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

    # show the plot
    plt.tight_layout()
    plt.show()
