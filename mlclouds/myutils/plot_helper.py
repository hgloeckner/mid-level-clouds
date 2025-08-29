def plot_cbar(fig, p, ax, var, varunit):
    fig.subplots_adjust(right=0.89)
    cax = fig.add_axes((0.9, 0.15, 0.01, 0.7))
    fig.colorbar(
        p,
        label=f"{var} / {varunit}",
        cax=cax,
        ax=ax,
        shrink=0.8,
        aspect=50,
        fraction=0.1,
    )

    return cax
