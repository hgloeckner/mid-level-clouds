# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from xhistogram.xarray import histogram

sys.path.append("../")
from myutils import open_datasets
from myutils import physics_helper as physics
import myutils.data_helper as dh
from orcestra import get_flight_segments

# %%
cid = "ipns://latest.orcestra-campaign.org"  # open_datasets.get_cid()
dropsondes = (
    xr.open_dataset(
        f"{cid}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
        engine="zarr",
    )
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)

lev4 = xr.open_dataset(
    f"{cid}/products/HALO/dropsondes/Level_4/PERCUSION_Level_4.zarr",
    engine="zarr",
)
# %%
colors = {
    "north": "#FF7982",
    "east": "#B6001E",
    "west": "#00b4d8",
}
# %%
wv, no_wv = open_datasets.open_wales(masked=False)
wv = wv.assign(q=physics.wv2q(wv))
no_wv = no_wv.sel(time=slice(np.datetime64("2024-08-10"), np.datetime64("2024-09-29")))
no_wv = xr.where(no_wv.bsrgl_flags == 8, 100, no_wv)
no_wv = no_wv.where((no_wv.bsrgl_flags == 0) | (no_wv.bsrgl_flags == 8)).sortby(
    "altitude"
)

# %%
wct = xr.DataArray(
    data=dh.find_highest_cloud_altitude(no_wv.sel(altitude=slice(200, None))),
    dims="time",
    name="cloud-top",
)
wct = wct.assign_coords(time=no_wv.time)
# %%
meta = get_flight_segments()
segments = sorted(
    [
        {
            **s,
        }
        for flight_id in meta.get("HALO", {}).keys()
        for s in meta.get("HALO", {}).get(flight_id, {}).get("segments", [])
        if "circle" in s["kinds"]
        if np.isin(s["segment_id"], lev4.circle_id.values)
    ],
    key=lambda s: s["start"],
)
# %%
lfrac = {}
mfrac = {}
hfrac = {}
ctmean = {}
for s in segments:
    ds = wct.sel(time=slice(s["start"], s["end"]))
    lfrac[s["segment_id"]] = ds.where(ds < 4000).count(dim="time") / ds.sizes["time"]
    ctmean[s["segment_id"]] = ds.mean(dim="time")
    mfrac[s["segment_id"]] = (
        ds.where((ds >= 4000) & (ds < 8000)).count(dim="time") / ds.sizes["time"]
    )
    hfrac[s["segment_id"]] = ds.where(ds >= 8000).count(dim="time") / ds.sizes["time"]
# %%
lcircle = lev4.swap_dims({"circle": "circle_id"}).assign(
    low_frac=("circle_id", [lfrac[cid] for cid in lev4.circle_id.values]),
    mid_frac=("circle_id", [mfrac[cid] for cid in lev4.circle_id.values]),
    high_frac=("circle_id", [hfrac[cid] for cid in lev4.circle_id.values]),
    ctmean=("circle_id", [ctmean[cid] for cid in lev4.circle_id.values]),
    cttmean=(
        "circle_id",
        [
            lev4.swap_dims({"circle": "circle_id"})
            .sel(circle_id=cid)
            .ta_mean.sel(altitude=ctmean[cid], method="nearest")
            .values
            for cid in lev4.circle_id.values
        ],
    ),
)
# %%
# %%
ds = dropsondes.dropna(dim="sonde", subset=["iwv"])
fig, ax = plt.subplots(figsize=(12, 5))
p = (
    histogram(
        ds.iwv,
        ds.altitude,
        bins=[np.arange(30, 70, 0.5), np.arange(0, 12000, 100)],
        weights=ds.rh,
    )
    / histogram(
        ds.iwv,
        ds.altitude,
        bins=[np.arange(30, 70, 0.5), np.arange(0, 12000, 100)],
    )
).plot(y="altitude_bin", cmap="BrBG", add_colorbar=False, ax=ax)
fig.colorbar(p, ax=ax, label="RH")

# %%


def calc_partial_omega(
    ds, intvar="div", pvar="p_mean", alt_dim="altitude", omega_name="omega"
):
    p_diff = xr.concat(
        [
            xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]}),
            ds[pvar]
            .where(~np.isnan(ds[intvar]), drop=True)
            .sortby(alt_dim)
            .diff(alt_dim),
        ],
        dim=alt_dim,
    )
    omega_diff = -ds[intvar] * p_diff
    omega_attrs = {
        "long_name": omega_name,
        "units": "Pa s$^{-1}$",
    }
    return ds.assign(
        {
            omega_name: (
                ds[intvar].dims,
                omega_diff.cumsum(alt_dim).broadcast_like(ds[intvar]).values,
                omega_attrs,
            )
        }
    )


omega_u = []
omega_v = []

for cid in lcircle.circle_id.values:
    ds = lcircle.sel(circle_id=cid)
    omega_u.append(calc_partial_omega(ds, intvar="u_dudx", omega_name="omega_u"))
    omega_v.append(calc_partial_omega(ds, intvar="v_dvdy", omega_name="omega_v"))
# %%
dspartial = xr.merge(
    [
        lcircle,
        xr.concat(omega_u, dim="circle_id"),
        xr.concat(omega_v, dim="circle_id"),
    ]
)
# %%

fig, ax = plt.subplots(figsize=(8, 6))

dspartial.isel(circle_id=5).omega_u.plot(y="altitude", ax=ax, label="omega_u")
dspartial.isel(circle_id=5).omega_v.plot(y="altitude", ax=ax, label="omega_v")
(dspartial.isel(circle_id=5).omega_u + dspartial.isel(circle_id=5).omega_v).plot(
    y="altitude", ax=ax, label="omega sum"
)
dspartial.isel(circle_id=5).omega.plot(y="altitude", ax=ax, label="omega")
ax.legend()
# %%


cvar = "omega"
tvar = "cttmean"
kwargs = dict(
    vmin=-20,  # -25,
    vmax=20,  # 25,
)
fig, axes = plt.subplots(figsize=(18, 15), nrows=3, sharey=True)
for idx, var in enumerate(["high_frac", "mid_frac", "low_frac"]):
    p = (
        histogram(
            dspartial[var],
            dspartial.ta_mean,
            bins=[np.arange(0, 1, 0.02), np.arange(210, 305, 1)],
            weights=dspartial[cvar] * 60 * 60 / 100,
        )
        / histogram(
            dspartial[var],
            dspartial.ta_mean,
            bins=[np.arange(0, 1, 0.02), np.arange(210, 305, 1)],
        )
    ).plot(
        y="ta_mean_bin", cmap="cmo.balance", add_colorbar=False, ax=axes[idx], **kwargs
    )

    (
        histogram(
            dspartial[var],
            bins=[np.arange(0, 1, 0.02)],
            weights=dspartial[tvar],
        )
        / histogram(
            dspartial[var],
            bins=[np.arange(0, 1, 0.02)],
        )
    ).plot(ax=axes[idx], color="k", marker="o", linestyle="")
axes[0].invert_yaxis()
axes[0].set_xlabel("fraction of high cloud in circle")
axes[1].set_xlabel("fraction of mid-level cloud in circle")
axes[2].set_xlabel("fraction of low cloud in circle")
for ax in axes:
    ax.axhline(273.15, color="k", linestyle="-", alpha=0.5)
    ax.set_ylabel("Mean Temperature / K")
sns.despine(offset={"left": 10})
fig.colorbar(p, ax=axes, label="omega / hPa hr$^{-1}$", fraction=0.1, shrink=0.5)

# %%
thres = 0.3
few_cond = (
    (dspartial.low_frac < thres)
    & (dspartial.mid_frac < thres)
    & (dspartial.high_frac < thres)
    & (dspartial.high_frac + dspartial.mid_frac + dspartial.low_frac < 0.3)
)
low_cond = (dspartial.ctmean < 4000) & (dspartial.low_frac >= thres)
mid_cond = (
    (dspartial.ctmean >= 4000)
    & (dspartial.ctmean < 8000)
    & (dspartial.mid_frac >= thres)
)
high_cond = (dspartial.ctmean >= 8000) & (dspartial.high_frac >= thres)
# %%
low = 2000
midlow = 5000
high = 8000
ds = lcircle
for name, cond in zip(
    ["no_clouds", "low clouds", "mid clouds", "high clouds", "all clouds"],
    [
        few_cond,
        low_cond,
        mid_cond,
        high_cond,
        (~few_cond) & (~low_cond) & (~mid_cond) & (~high_cond),
    ],
):
    print(name)
    ds = dspartial.where(cond, drop=True)
    print(ds.sizes)
    print(ds.dropna(dim="circle_id", how="all", subset=["omega"]).sizes)
    print("number of circles:", ds.sizes["circle_id"])
    ds = ds.mean("circle_id")
    print(
        "high | dudx | ",
        ds.u_dudx.sel(altitude=slice(high, None)).mean("altitude").values,
        "\n",
    )
    print(
        "high | dvdy |",
        ds.v_dvdy.sel(altitude=slice(high, None)).mean("altitude").values,
        "\n",
    )
    print(
        "high | div |",
        ds.div.sel(altitude=slice(high, None)).mean("altitude").values,
        "\n",
    )

    print(
        "mid-high | dudx | ",
        ds.u_dudx.sel(altitude=slice(midlow, high)).mean("altitude").values,
        "\n",
    )
    print(
        "mid-high | dvdy |",
        ds.v_dvdy.sel(altitude=slice(midlow, high)).mean("altitude").values,
        "\n",
    )
    print(
        "mid-high | div |",
        ds.div.sel(altitude=slice(midlow, high)).mean("altitude").values,
        "\n",
    )

    print(
        "mid-low | dudx | ",
        ds.u_dudx.sel(altitude=slice(low, midlow)).mean("altitude").values,
        "\n",
    )
    print(
        "mid-low | dvdy |",
        ds.v_dvdy.sel(altitude=slice(low, midlow)).mean("altitude").values,
        "\n",
    )
    print(
        "mid-low | div |",
        ds.div.sel(altitude=slice(low, midlow)).mean("altitude").values,
        "\n",
    )

    print(
        "low | dudx | ",
        ds.u_dudx.sel(altitude=slice(0, low)).mean("altitude").values,
        "\n",
    )
    print(
        "low | dvdy |",
        ds.v_dvdy.sel(altitude=slice(0, low)).mean("altitude").values,
        "\n",
    )
    print(
        "low | div |", ds.div.sel(altitude=slice(0, low)).mean("altitude").values, "\n"
    )

# %%


cvar = "omega"
tvar = "cttmean"
variables = ["high_frac", "mid_frac", "low_frac"]
kwargs = dict(
    vmin=-20,  # -25,
    vmax=20,  # 25,
)
sns.set_context("paper", font_scale=0.8)
cw = 190 / 25.4
fig, axs = plt.subplots(
    figsize=(cw, 2 / 3 * cw),
    ncols=2,
    width_ratios=[0.2, 0.8],
    nrows=3,
    sharex="col",
    sharey=True,
)
for idx, cond in enumerate([high_cond, mid_cond, low_cond]):
    axes = axs[:, 1]
    var = variables[idx]
    ds = dspartial.where(cond, drop=True)
    p = (
        histogram(
            ds[var],
            ds.ta_mean,
            bins=[np.arange(0, 1, 0.02), np.arange(210, 305, 1)],
            weights=ds[cvar] * 60 * 60 / 100,
        )
        / histogram(
            ds[var],
            ds.ta_mean,
            bins=[np.arange(0, 1, 0.02), np.arange(210, 305, 1)],
        )
    ).plot(
        y="ta_mean_bin", cmap="cmo.balance", add_colorbar=False, ax=axes[idx], **kwargs
    )

    (
        histogram(
            ds[var],
            bins=[np.arange(0, 1, 0.02)],
            weights=ds[tvar],
        )
        / histogram(
            ds[var],
            bins=[np.arange(0, 1, 0.02)],
        )
    ).plot(ax=axes[idx], color="k", marker="o", linestyle="", markersize=2)
for idx, (cond, lim) in enumerate(
    zip([high_cond, mid_cond, low_cond], [None, 230, 230])
):
    for cvar, ls, label in zip(
        ["omega", "omega_u", "omega_v"],
        ["-", "--", ":"],
        ["$\omega$", "$\omega_u$", "$\omega_v$"],
    ):
        axes = axs[:, 0]
        cloudds = dspartial.where(cond, drop=True)
        (
            histogram(
                cloudds.ta_mean,
                bins=[np.arange(210, 305, 1)],
                weights=cloudds[cvar] * 60 * 60 / 100,
            )
            / histogram(
                cloudds.ta_mean,
                bins=[np.arange(210, 305, 1)],
            )
        ).rolling(
            ta_mean_bin=5,
            center=True,
        ).mean().sel(ta_mean_bin=slice(lim, None)).plot(
            y="ta_mean_bin", ax=axes[idx], label=label, linestyle=ls, color="k"
        )


axs[1, 0].legend()
axes[0].invert_yaxis()
for ax in axs.flatten():
    ax.axhline(273.15, color="k", linestyle="-", alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
for ax in axs[:, 0]:
    ax.axvline(0, color="k", linestyle="-", alpha=0.5)
    ax.set_ylabel("Temperature / K")
axs[2, 0].set_xlabel("omega / hPa hr$^{-1}$")
axs[0, 1].set_xlim(0.3, 1)
axs[0, 0].set_xlim(-10, 4)
axs[0, 1].set_xlabel("fraction of high cloud in high-cloud circles")
axs[1, 1].set_xlabel("fraction of mid-level cloud in mid-level cloud circles")
axs[2, 1].set_xlabel("fraction of low cloud in low-cloud circles")
sns.despine(offset=5)
fig.tight_layout()
fig.colorbar(
    p,
    ax=axs[:, 1],
    label="omega / hPa hr$^{-1}$",
    extend="both",
    fraction=0.1,
    shrink=0.5,
)

fig.savefig("/scratch/m/m301046/omega_cloudtype.pdf")
