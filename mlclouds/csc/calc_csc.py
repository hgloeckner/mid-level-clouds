# %%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import myutils.physics_helper as ph
import matplotlib.lines as mlines

levante = True
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
arts["real"] = (
    arts["real"]
    .drop_vars(["interpolated_time"])
    .rolling(altitude=20, center=True)
    .mean()
)


# %%
def calc_cs_convergence(ds, ta_var="ta", qvar="q"):
    ds = ds.assign(
        stab=ph.get_stability(ds.theta, ds[ta_var]),
        rho=ph.density_from_q(ds.p, ds[ta_var], ds[qvar]),
    )
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
cvselegend = [
    mlines.Line2D(
        [],
        [],
        color="k",
        label="C-shape",
    ),
    mlines.Line2D(
        [],
        [],
        color="k",
        linestyle=":",
        label="E-shape",
    ),
]

radlegend = [
    mlines.Line2D(
        [],
        [],
        color="#005652",
        label="LW only",
    ),
    mlines.Line2D(
        [],
        [],
        color="k",
        label="LW + mean(SW)",
    ),
    mlines.Line2D(
        [],
        [],
        color="#f47721",
        label="LW + max(SW)",
    ),
]

lowerlegend = [
    mlines.Line2D(
        [],
        [],
        color="k",
        label="real $T$ and RH (low)",
    ),
    mlines.Line2D(
        [],
        [],
        color="k",
        linestyle=":",
        label="moist adiabat, E-shape (low)",
    ),
]


# %%
key = "adiabat"
htgrs = [("_lw", "#005652"), ("", "k"), ("_sw_high", "#f47721")]
sondes = [
    ("adiabat", "clow", "-", 0),
    ("adiabat", "elow", ":", 0),
    ("real", "elow", "-", 1),
    ("adiabat", "elow", ":", 1),
]
cw = 8.27
sns.set_context("paper")
fig, axes = plt.subplots(
    ncols=3, nrows=2, figsize=(cw, cw * 2 / 3), sharey=True, sharex="col"
)


for key, sonde, ls, idx in sondes:
    ds = cscarts[key].sel(sonde=sonde).sel(altitude=slice(0, 12000))
    for htgr, color in htgrs:
        axes[idx, 0].plot(ds[f"htgr{htgr}"] * 60 * 60 * 24, ds.ta, ls=ls, color=color)
        axes[idx, 1].plot(
            ds[f"csc_cool{htgr}"] * 60 * 60 * 24, ds.ta, ls=ls, color=color
        )
        axes[idx, 2].plot(
            (ds[f"csc_cool{htgr}"] + ds[f"csc_stab{htgr}"]) * 60 * 60 * 24,
            ds.ta,
            ls=ls,
            color=color,
        )


axes[0, 0].invert_yaxis()
axes[0, 0].set_ylim(290, 250)
axes[0, 0].set_xlim(-3, None)
for ax in axes[0, 1:]:
    ax.set_xlim(-0.3, 0.26)
    ax.set_xticks([-0.25, 0, 0.25])
for ax in axes[:, 0]:
    ax.set_ylabel("$T$ / K")
    ax.set_yticks([290, 280, 270, 260, 250])
for ax in axes.flatten():
    ax.axhline(273.15, color="grey", ls="-", linewidth=1)
    ax.axvline(0, color="grey", ls="-", linewidth=1)
axes[1, 0].set_xlabel(r"$\mathcal{H}$ / K day$^{-1}$")
axes[1, 1].set_xlabel(r"$\mathrm{CSC}_\mathcal{H}$ / day$^{-1}$")
axes[1, 2].set_xlabel(
    r"$\mathrm{CSC}_\mathcal{H} + \mathrm{CSC}_\mathcal{\sigma}$ / day$^{-1}$"
)


sns.despine(offset=5)
fig.tight_layout()
axes[0, 1].legend(
    bbox_to_anchor=(-0.3, 0.975),
    ncol=1,
    loc="upper left",
    framealpha=1,
    handles=cvselegend,
    fontsize=8,
)

axes[0, 2].legend(
    bbox_to_anchor=(-0.3, 0.975),
    ncol=1,
    loc="upper left",
    framealpha=1,
    handles=radlegend,
    fontsize=8,
)
axes[1, 1].legend(
    bbox_to_anchor=(-0.5, 0.975),
    ncol=1,
    loc="upper left",
    framealpha=1,
    handles=lowerlegend,
    fontsize=8,
)
# %%

cvselegend = [
    mlines.Line2D(
        [],
        [],
        color="k",
        label="C-shape",
    ),
]
"""
        mlines.Line2D(
                    [],
                    [],
                    color="k",
                    linestyle=":",
                    label="E-shape",
                ), 
]
"""

radlegend = [
    mlines.Line2D(
        [],
        [],
        color="#005652",
        label="LW only",
    ),
]
"""             
mlines.Line2D(
            [],
            [],
            color="k",
            label="LW + mean(SW)",
        ), 

]
"""
htgrs = [  # ("", "k", "adiabat", "elow", ":", 0),
    ("_lw", "#005652", "adiabat", "clow", "-", 0),
    # ("_lw", "#005652", "adiabat", "elow", ":", 0),
]  # ,, ("_sw_high", "#f47721")]
cw = 8.27
sns.set_context("talk")
fig, axes = plt.subplots(ncols=3, figsize=(12, 5), sharey=True)

"""
"""
for htgr, color, key, sonde, ls, idx in htgrs:
    ds = cscarts[key].sel(sonde=sonde).sel(altitude=slice(0, 12000))

    axes[0].plot(ds[f"htgr{htgr}"] * 60 * 60 * 24, ds.ta, ls=ls, color=color)
    axes[1].plot(ds[f"csc_cool{htgr}"] * 60 * 60 * 24, ds.ta, ls=ls, color=color)
    axes[2].plot(
        (ds[f"csc_cool{htgr}"] + ds[f"csc_stab{htgr}"]) * 60 * 60 * 24,
        ds.ta,
        ls=ls,
        color=color,
    )

axes[0].set_xticks([-3, -2, -1])
axes[0].set_xlim(-3, -0.5)
axes[0].set_ylim(290, 250)
axes[0].set_xlim(-3, None)
axes[0].set_ylabel("temperature / K")
axes[0].set_yticks([290, 280, 270, 260, 250])
for ax in axes[1:]:
    ax.set_xlim(-0.2, 0.2)
    ax.set_xticks([-0.2, 0, 0.2])
for ax in axes:
    ax.axhline(273.15, color="grey", ls="-", linewidth=1)
    ax.axvline(0, color="grey", ls="-", linewidth=1)
axes[0].set_xlabel(r"heating rate / K day$^{-1}$")
axes[1].set_xlabel("clear-sky convergence \n (heating rate)" + "\n" + "/ day$^{-1}$")
axes[2].set_xlabel("clear-sky convergence \n\n / day$^{-1}$")


sns.despine(offset=5)
fig.tight_layout()
"""
"""
axes[1].legend(
    bbox_to_anchor=(-0.3, 0.995),
    ncol=1,
    loc="upper left",
    framealpha=1,
    handles=cvselegend,
    fontsize=16,
)

axes[2].legend(
    bbox_to_anchor=(-0.5, 0.995),
    ncol=1,
    loc="upper left",
    framealpha=1,
    handles=radlegend,
    fontsize=16,
)
fig.savefig(scratchpath + "csc_cshape.pdf")

# %%

# %%
cw = 8.27
sns.set_context("talk")
fig, ax = plt.subplots()
"""
"""
for sonde, ls, label, alpha, color in [
    ("reallow", "-", "real", 1, "#005652"),
    ("clow", "-", "C-shape", 0.5, "#005652"),
    ("elow", ":", "E-shape", 0.5, "#005652"),
]:
    ds = arts["adiabat"].sel(sonde=sonde, altitude=slice(0, 15000))
    ax.plot(
        ds.rh * 100,
        ds.ta,
        ls=ls,
        label=label,
        alpha=alpha,
        c=color,
    )
ax.set_ylabel("temperature / K")
ax.set_xlabel("relative humidity / %")
ax.invert_yaxis()
ax.set_ylim(305, 210)
ax.set_xlim(30, 90)
ax.legend()
sns.despine(offset=10)
fig.tight_layout()
fig.savefig(scratchpath + "rh_title.pdf")
# %%
cw = 8.27
sns.set_context("talk")
fig, ax = plt.subplots()
"""
"""
htgrs = [
    ("", "k", "adiabat", "elow", ":", "E-shape, LW and SW"),
    ("_lw", "#005652", "adiabat", "clow", "-", "C-shape, LW"),
    ("_lw", "#005652", "adiabat", "elow", ":", "E-shape, LW"),
]

for htgr, color, key, sonde, ls, label in htgrs:
    ds = arts["adiabat"].sel(sonde=sonde, altitude=slice(0, 15000))
    ax.plot(ds[f"htgr{htgr}"] * 60 * 60 * 24, ds.ta, ls=ls, color=color, label=label)
ax.set_ylabel("temperature / K")
"""
ax.axvline(
    -1.64, (290-287)/40, (290-273)/40, color="grey", lw=1, 

)
"""
# ax.set_xlabel("relative humidity / %")
ax.invert_yaxis()
ax.set_ylim(290, 250)
ax.set_xlim(-3, -0.5)
ax.legend()
sns.despine(offset=10)
fig.tight_layout()
fig.savefig(scratchpath + "htgr_full.pdf")

# %%
import myutils.open_datasets as od
import myutils.physics_helper as ph
import myutils.data_helper as dh

beach = (
    od.open_dropsondes(od.get_cids()["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)
