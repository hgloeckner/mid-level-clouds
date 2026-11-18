# %%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import myutils.physics_helper as ph
import matplotlib.lines as mlines
import moist_thermodynamics.constants as mtc
import orcestra
import numpy as np

# %%


def calc_stability(ds, ta_var="ta", qvar="q"):
    ds = ds.assign(
        stab=ph.get_stability(ds.theta, ds[ta_var]),
        rho=ph.density_from_q(ds.p, ds[ta_var], ds[qvar]),
    )
    return ds


levante = False
if levante:
    datapath = "/work/mh0066/m301046/data/mlclouds/"
    scratchpath = "/scratch/m/m301046/"
else:
    datapath = "/Users/helene/Documents/data/mlclouds/"
    scratchpath = "/Users/helene/Documents/scratch/"
idealized = {
    key: xr.open_dataset(datapath + f"idealized_{key}_profiles.nc")
    for key in ["adiabat", "real"]
}

arts = {
    key: xr.open_dataset(
        datapath + f"idealized_{key}_arts.zarr", engine="zarr", chunks={}
    )
    .load()
    .rename(
        {
            "sw_heating_rate": "htgr_sw",
            "lw_heating_rate": "htgr_lw",
        }
    )
    for key in ["adiabat", "real"]
}
arts = {
    key: ds.assign(
        htgr_sw=ds.htgr_sw.fillna(0) / (60 * 60 * 24),
        htgr_lw=ds.htgr_lw.fillna(0) / (60 * 60 * 24),
    ).assign(htgr_sw_mean=ds.htgr_sw.fillna(0).mean("hour_of_day") / (60 * 60 * 24))
    for key, ds in arts.items()
}
arts = {
    key: xr.merge([ds, idealized[key]], compat="override").assign(
        htgr=ds.htgr_lw + ds.htgr_sw.mean("hour_of_day"),
        htgr_sw_high=ds.htgr_lw + ds.htgr_sw.sel(hour_of_day=12),
    )
    for key, ds in arts.items()
}

arts = {key: calc_stability(ds) for key, ds in arts.items()}
arts["realroll"] = (
    arts["real"]
    .drop_vars(["interpolated_time"])
    .rolling(altitude=20, center=True)
    .mean()
)


# %%


def calc_cs_convergence(ds, ta_var="ta", qvar="q"):
    for htgr in [
        (ds.htgr, ""),
        (ds.htgr_lw, "_lw"),
        (ds.htgr_sw_mean, "_sw_mean"),
        (ds.htgr_sw_high, "_sw_high"),
    ]:
        ds = ds.assign(
            {
                f"csc_stab{htgr[1]}": ph.get_csc_stab(ds.rho, ds.stab, htgr[0]),
                f"csc_cool{htgr[1]}": ph.get_csc_cooling(ds.rho, ds.stab, htgr[0]),
                f"mass_flux{htgr[1]}": ph.mass_flux(ds.rho, ds.stab, htgr[0]),
            }
        )
    return ds


cscarts = {key: calc_cs_convergence(ds) for key, ds in arts.items()}


# %%
cw = 8.27
sns.set_context("talk")
fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
"""
"""
for sonde, ls, label, alpha, color in [
    ("reallow", "-", "real", 1, "#005652"),
    ("clow", "-", "C-shape", 0.5, "#005652"),
    ("elow", ":", "E-shape", 0.5, "#005652"),
]:
    ds = arts["adiabat"].sel(sonde=sonde, altitude=slice(0, 15000))
    axes[0].plot(
        ds.rh * 100,
        ds.ta,
        ls=ls,
        label=label,
        alpha=alpha,
        c=color,
    )
axes[0].set_ylabel("$T$ / K")
axes[0].set_xlabel("RH / %")
axes[0].invert_yaxis()
axes[0].set_ylim(300, 210)
axes[0].set_xlim(30, 90)
axes[0].legend()

for key, alpha in [("adiabat", 0.5), ("real", 1)]:
    axes[1].plot(
        cscarts[key].stab.mean("sonde") * 1000,
        cscarts[key].ta.mean("sonde"),
        label=key,
        color="#0B257A",
        alpha=alpha,
    )

axes[1].set_xlabel("$\sigma$ / K km$^{-1}$")
axes[1].legend()
axes[1].set_ylim(300, 210)

htgrs = [
    ("", "#005652", "real", "clow", "-", "C-shape, real $T$", 1),
    ("", "#005652", "adiabat", "elow", ":", "E-shape, adiabat", 0.5),
    ("", "#005652", "adiabat", "clow", "-", "C-shape, adiabat", 0.5),
    ("_lw", "k", "adiabat", "clow", "-", "C-shape, adiabat, LW", 0.5),
]

for htgr, color, key, sonde, ls, label, alpha in htgrs:
    ds = arts[key].sel(sonde=sonde, altitude=slice(0, 15000))
    axes[2].plot(
        ds[f"htgr{htgr}"] * 60 * 60 * 24,
        ds.ta,
        ls=ls,
        color=color,
        label=label,
        alpha=alpha,
    )

axes[2].set_xlabel("$\mathcal{H}$ / K day$^{-1}$")
axes[2].set_ylim(300, 210)
axes[2].set_xlim(-2.7, -0.7)
axes[2].legend()
axes[2].axvline(-1.64, (290 - 287) / 40, (290 - 273) / 40, color="grey", lw=1, ls=":")
sns.despine(offset=10)

fig.tight_layout()
fig.savefig(scratchpath + "rh_title.pdf")

# %%

fig, axes = plt.subplots(ncols=3, figsize=(12, 5), sharey=True, sharex=True)
htgrs = [
    ("", "#005652", "adiabat", "clow", "adiabat, C-shape"),
    ("", "#005652", "adiabat", "elow", "adiabat, E-shape"),
    ("", "#005652", "realroll", "elow", "real $T$, E-shape"),
]

for idx, htgr in enumerate(htgrs):
    htgrds = cscarts[htgr[2]].sel(sonde=htgr[3], altitude=slice(0, 15000))
    for pltds, label, ls, alpha in [
        (htgrds["csc_stab" + htgr[0]], "CSC$_{\sigma}$", ":", 0.5),
        (htgrds["csc_cool" + htgr[0]], "CSC$_{\mathcal{H}}$", "--", 0.5),
        (htgrds["csc_cool" + htgr[0]] + htgrds["csc_stab" + htgr[0]], "CSC", "-", 1),
    ]:
        axes[idx].plot(
            pltds * 60 * 60 * 24,
            htgrds.ta,
            color=htgr[1],
            label=label,
            ls=ls,
            alpha=alpha,
        )
    axes[idx].set_title(htgr[4])
    axes[idx].axvline(  #
        0, color="grey", lw=1, ls="-"
    )
    axes[idx].axhline(273.15, color="grey", lw=1, ls="-")
    axes[idx].set_xlabel("CSC / day$^{-1}$")
axes[0].invert_yaxis()
axes[0].set_ylim(285, 250)
axes[0].set_xlim(-0.15, 0.15)
axes[0].set_ylabel("$T$ / K")
axes[0].legend(fontsize=12)

sns.despine()
fig.tight_layout()
# fig.savefig(scratchpath + "csc_full.pdf")
# %%
