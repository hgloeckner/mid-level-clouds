# %%

import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
# %%

sonde_data = xr.open_dataset("/work/mh0066/m301046/data/mlclouds/idealized_profiles.nc")

# %%
sns.set_context("talk", font_scale=0.8)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))


for sonde, alpha, ls, label in [
    ("reallow", 1, "-", "BEACH clear-sky"),
    ("clow", 0.5, ":", "C-shape"),
    # ("elow", 0.5, "-", "E-shape"),
]:
    plt_data = sonde_data.sel(sonde=sonde)
    ax.plot(
        plt_data.rh * 100,
        plt_data.T,
        alpha=alpha,
        color="#005555",
        linestyle=ls,
        label=label,
    )

ax.invert_yaxis()
ax.legend()
ax.set_xlim(0, 100)
ax.set_xlabel("relative humidity / %")
ax.set_ylabel("temperature / K")
ax.set_yticks([300, 280, 260, 240, 220], labels=["300", "280", "260", "240", "220"])
ax.tick_params(size=5, width=1)
ax.set_ylim(None, 220)
ax.spines["bottom"].set_linewidth(1)
ax.spines["left"].set_linewidth(1)
sns.despine(offset={"bottom": 5})
fig.savefig("../plots/idealized_rh_profiles_c.pdf", bbox_inches="tight")
