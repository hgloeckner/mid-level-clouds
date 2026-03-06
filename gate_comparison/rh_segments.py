# %%
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from xhistogram.xarray import histogram

import sys

sys.path.append("../")
import myutils.open_datasets as opends  # noqa
import myutils.data_helper as dh  # noqa
import myutils.plot_helper as ph  # noqa

# %%
radios = opends.open_radiosondes(
    "bafybeigensqyqxfyaxgyjhwn6ytdpi3i4sxbtffd4oc27zbimyro4hygjq"
)


drops = opends.open_dropsondes(
    "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
)

gate = opends.open_gate("QmWmYbYbW51bpYGREctj1LLWSMrPc7sEXkgDzhsDYsW3qg")

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)

datasets = {
    "orcestra-up": radios.where(radios.ascent_flag == 1),
    "orcestra-down": xr.concat(
        [radios.where(radios.ascent_flag == 0)],
        dim="sonde_id",  # drops,
    ),
    "gate": gate,
}
# %%

varmin = 0.95

segments = {}
for key, ds in datasets.items():
    segments[key] = dh.get_segments(
        ds=ds.where(ds.launch_lon > -40, drop=True),
        var="rh",
        varmin=varmin,
    )

lengths = {}
starts = {}
ends = {}
for key, di in segments.items():
    lengths[key] = list(
        itertools.chain.from_iterable([val[0] for _, val in di.items()])
    )
    starts[key] = list(itertools.chain.from_iterable([val[1] for _, val in di.items()]))
    ends[key] = list(itertools.chain.from_iterable([val[2] for _, val in di.items()]))


# %%
plotvar = "bottom"
plotdict = {
    "depths": (lengths, (0, 10)),
    "bottom": (starts, (255, 305)),
    "top": (ends, (255, 305)),
}

fig, ax = plt.subplots(figsize=(6, 6))
for key, val in plotdict[plotvar][0].items():
    hist = np.histogram(val, bins=100, range=plotdict[plotvar][1])
    xvals = (hist[1][1:] + hist[1][:-1]) / 2
    yvals = hist[0] / datasets[key].sizes["sonde_id"]
    ax.step(
        x=xvals,
        y=yvals,
        label=key,
    )
    ax.fill_between(xvals, 0, yvals, step="pre", alpha=0.2)
ax.set_xlabel(f"{plotvar} / K")
ax.set_ylabel("count normalized by number of sondes")
ax.set_title(f"RH > {varmin}, segment {plotvar} ")
ax.set_xlim(plotdict[plotvar][1])
ax.set_ylim(0, None)
ax.legend()
sns.despine()
# %%


hists = {}
for key in ["orcestra-up", "orcestra-down", "gate"]:
    ds = xr.Dataset(
        data_vars={
            "depth": ("x", lengths[key]),
            "start": ("x", starts[key]),
            "weights": ("x", np.full(len(lengths[key]), 1 / len(lengths[key]))),
        }  # 1/ datasets[key].sizes["sonde_id"
    )
    print(ds["weights"].values)
    hists[key] = histogram(
        ds["start"],
        ds["depth"],
        bins=[np.linspace(255, 302, 50), np.linspace(0, 10, 50)],
        dim="x",
        weights=ds["weights"],
    )
# %%

fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
for i, key in enumerate(["orcestra-up", "orcestra-down", "gate"]):
    p = hists[key].plot(ax=axes[i], vmax=0.0015, add_colorbar=False, cmap="cmo.ice")
    axes[i].set_title(key)

    # axes[i].invert_yaxis()

ph.plot_cbar(fig, p, ax, "count weighted by number segments", "1")

# %%

# %%
for i, key in enumerate(["orcestra-up", "orcestra-down", "gate"]):
    hists[key].mean("depth_bin").plot(label=key)
plt.legend()
# %%
