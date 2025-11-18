# %%
import seaborn as sns
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import cmocean as cmo
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import lam_orcestra.helper as help
import myutils.data_helper as dh
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
def get_cids():
    orcestra_main = "QmXkSUDo97PaDxsPzCPXJXwCFDLBMp7AVdPdV5CBQoagUN"
    return {
        "gate": "QmWFfuLW7VSqEFrAwaJr1zY9CzWqF4hC22yqgXELmY133K",
        "orcestra": orcestra_main,
        "radiosondes": f"{orcestra_main}/products/Radiosondes/Level_2/RAPSODI_RS_ORCESTRA_level2.zarr",
        "dropsondes": f"{orcestra_main}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    }


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return ds.reset_coords(["launch_altitude"])


def get_hist_of_ta_2d(
    da_t, da_var, var_binrange, ta_binrange=(240, 305), var_nb=100, ta_nb=200
):
    bins_ta = np.linspace(ta_binrange[0], ta_binrange[1], ta_nb)
    bins_var = np.linspace(var_binrange[0], var_binrange[1], var_nb)
    return histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=da_t.dims)


def rolling_hist(ds, ta_bin_low=15, rh_bin_low=5, ta_bin_up=20, rh_bin_up=7):
    return xr.concat(
        [
            (
                ds.where(ds > 0)
                .rolling(ta_bin=ta_bin_low, center=True, min_periods=3)
                .sum()
                .rolling(rh_bin=rh_bin_low, center=True, min_periods=3)
                .sum()
                .sel(ta_bin=slice(270, 305))
            ),
            (
                ds.where(ds > 0)
                .rolling(ta_bin=ta_bin_up, center=True, min_periods=3)
                .sum()
                .rolling(rh_bin=rh_bin_up, center=True, min_periods=3)
                .sum()
                .sel(ta_bin=slice(None, 270))
            ),
        ],
        dim="ta_bin",
    ).sortby("ta_bin")


def interpolate_gaps(ds):
    akima_vars = ["u", "v"]
    linear_vars = ["theta", "q", "p"]

    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="akima", max_gap=1500)
            for var in akima_vars
        }
    )
    ds = ds.assign(p=np.log(ds.p))
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="linear", max_gap=1500)
            for var in linear_vars
        }
    )
    ds = ds.assign(p=np.exp(ds.p))
    ds = ds.assign(
        ta=mtf.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mtf.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )

    return ds


def extrapolate_sfc(ds):
    """
    Extrapolate surface values to the lowest level.
    This function assumes that the dataset has an altitude dimension.
    """
    constant_vars = ["u", "v", "theta", "q"]
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(
                dim="altitude", method="nearest", max_gap=300, fill_value="extrapolate"
            )
            for var in constant_vars
        }
    )
    ds = ds.assign(
        p=np.exp(
            np.log(ds.p).interpolate_na(
                dim="altitude", method="linear", max_gap=300, fill_value="extrapolate"
            )
        )
    )
    ds = ds.assign(
        ta=mtf.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mtf.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )
    return ds


# %%

lam_sondes = xr.open_dataset(
    "/scratch/m/m301046/lam_sondes_z.zarr",
    engine="zarr",
)
ifs = xr.open_dataset(
    "/work/mh0492/m301067/orcestra/results/timeseries/ifs_interpolated_on_dropsondes_profiles_2nd-days.nc"
)

cids = get_cids()
beach = (
    open_dropsondes(cids["dropsondes"])
    .pipe(interpolate_gaps)
    .pipe(extrapolate_sfc)
    .isel(sonde=slice(2, None))
)

gate = (
    (
        xr.open_dataset(
            "ipfs://QmWZryTDTZu68MBzoRDQRcUJzKdCrP2C4VZfZw1sZWMJJc", engine="zarr"
        )
        .set_coords(["launch_lat", "launch_lon", "launch_time"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=slice("1974-08-10", "1974-09-30"))
        .swap_dims({"launch_time": "sonde"})
    )
    .pipe(interpolate_gaps)
    .pipe(extrapolate_sfc)
)
# %% create lam dataset
lam_ds = lam_sondes.where(
    lam_sondes.sonde_id.isin(beach.sonde_id.values),
    drop=True,
)
altdim = "z"

lam_subset = lam_ds.sortby(altdim, ascending=False).reset_coords(
    ["launch_lat", "launch_lon"]
)
lam_subset = lam_subset.assign(
    rh=mtf.specific_humidity_to_relative_humidity(
        q=lam_subset.qv,
        p=lam_subset.pfull,
        T=lam_subset.ta,
        es=svp.liq_wagner_pruss,
    ),
    ice_rh=mtf.specific_humidity_to_relative_humidity(
        q=lam_subset.qv,
        p=lam_subset.pfull,
        T=lam_subset.ta,
        es=svp.ice_wagner_etal,
    ),
    theta=mtf.theta(lam_subset.ta, lam_subset.pfull),
).sel(z=slice(15000, None))
lam_subset = lam_subset.assign(
    n2=help.apply_brunt_vaisala_frequency(lam_subset, altdim=altdim, q="qv")
).rename({"z": "altitude"})

# %%
ifs = ifs.assign(
    rh=mtf.specific_humidity_to_relative_humidity(
        q=ifs.q,
        p=ifs.pressure,
        T=ifs.t,
        es=svp.liq_wagner_pruss,
    ),
    theta=mtf.theta(ifs.t, ifs.pressure),
).rename({"t": "ta", "height": "altitude", "sonde_lat": "launch_lat"})
ifs = ifs.assign(n2=help.apply_brunt_vaisala_frequency(ifs, altdim="altitude", q="q"))

beach = beach.assign(
    n2=help.apply_brunt_vaisala_frequency(beach, altdim="altitude", q="q"),
    ice_rh=mtf.specific_humidity_to_relative_humidity(
        q=beach.q,
        p=beach.p,
        T=beach.ta,
        es=svp.ice_wagner_etal,
    ),
).reset_coords(["launch_lat", "launch_lon"])

gate = gate.assign(
    n2=help.apply_brunt_vaisala_frequency(gate, altdim="altitude", q="q"),
    ice_rh=mtf.specific_humidity_to_relative_humidity(
        q=gate.q,
        p=gate.p,
        T=gate.ta,
        es=svp.ice_wagner_etal,
    ),
)
# %%
Px = 100900.0
P = np.arange(Px, 4000.0, -500)
T2 = 301
q2 = 0.018  # 9182267570514704

pseudo2 = help.make_sounding_from_adiabat(P, T2, q2, thx=mtf.theta_e_bolton, Tmin=195)
pseudo = pseudo2.assign(
    q=mtf.relative_humidity_to_specific_humidity(
        RH=beach.mean("sonde").rh.interp(altitude=pseudo2.altitude),
        p=pseudo2.P,
        T=pseudo2.T,
        es=svp.liq_wagner_pruss,
    )
)

n2_pseudo = mtf.brunt_vaisala_frequency(
    pseudo2["theta"], pseudo2["q"], pseudo2["altitude"]
)

# %% calc RH histograms
ta_binrange = (210, 305)
rh_binrange = (0, 1.1)
ta_nb = 200
rh_nb = 100


rh_histograms = {}
ice_histograms = {}
n2_hist = {}
for ds, name in [
    (lam_subset, "lam-total"),
    (ifs, "ifs-total"),
    (beach, "beach-total"),
    (dh.sel_sub_domain(beach, dh.gate_a), "beach-gate"),
    (dh.sel_sub_domain(beach, dh.east), "beach-east"),
    (dh.sel_sub_domain(beach, dh.west), "beach-west"),
    (dh.sel_sub_domain(beach, dh.north), "beach-north"),
    (dh.sel_sub_domain(gate, dh.gate_a), "gate"),
    (dh.sel_sub_domain(lam_subset, dh.east), "lam-east"),
    (dh.sel_sub_domain(lam_subset, dh.west), "lam-west"),
    (dh.sel_sub_domain(lam_subset, dh.north), "lam-north"),
]:
    ds = ds.sortby("altitude").sel(altitude=slice(0, 14000))
    rh_histograms[name] = histogram(
        ds.rh,
        ds.ta,
        bins=[np.linspace(0, 1.1, 100), np.linspace(210, 305, 200)],
    ).compute()
    try:
        ice_histograms[name] = histogram(
            ds.ice_rh,
            ds.ta,
            bins=[np.linspace(0, 1.1, 100), np.linspace(210, 305, 200)],
        ).compute()
    except AttributeError:
        pass
    try:
        n2_hist[name] = histogram(
            ds.n2,
            ds.ta,
            bins=[np.linspace(0, 5e-2, 100), np.linspace(240, 305, 65)],
        ).compute()
    except AttributeError:
        pass
# %%

kwargs = dict(
    lam={
        "contour": dict(
            cmap="Oranges", add_colorbar=False
        ),  # sns.light_palette("darkorange",n_colors=7, as_cmap=True)),
        "line": dict(c="firebrick", linewidth=4),
    },
    beach={
        "contour": dict(cmap="cmo.ice_r", linestyles="dotted", linewidths=3),
        "line": dict(c="royalblue", linewidth=4, linestyle="--"),
    },
)

levels = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
sns.set_context("talk", font_scale=0.9)
fig, ax = plt.subplots(figsize=(6, 5.5))
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.6)

region = "total"
for name, label in [(f"lam-{region}", "ICON"), (f"beach-{region}", "Dropsondes")]:
    plthist = xr.concat(
        [
            (
                (ice_histograms[name] / ice_histograms[name].sum("ice_rh_bin"))
                .where(ice_histograms[name] > 0)
                .sel(ta_bin=slice(None, 273))
                .rolling(ta_bin=20, center=True, min_periods=3)
                .sum()
                .rolling(ice_rh_bin=7, center=True, min_periods=3)
                .sum()
            ).rename({"ice_rh_bin": "rh_bin"}),
            (
                (rh_histograms[name] / rh_histograms[name].sum("rh_bin"))
                .where(rh_histograms[name] > 0)
                .sel(ta_bin=slice(273, None))
                .rolling(ta_bin=15, center=True, min_periods=3)
                .sum()
                .rolling(rh_bin=7, center=True, min_periods=3)
                .sum()
            ),
        ],
        dim="ta_bin",
    )
    plthist = plthist / plthist.sum(("rh_bin"))

    if "beach" in name:
        fct = plthist.plot.contour
    else:
        fct = plthist.plot.contourf
    """
    """
    p = fct(
        y="ta_bin",
        ax=ax,
        levels=levels,
        **kwargs[name.split("-")[0]]["contour"],
    )

    (plthist * plthist.rh_bin).sum("rh_bin").sel(ta_bin=slice(None, 303)).plot(
        y="ta_bin",
        ax=ax,
        label=label,
        **kwargs[name.split("-")[0]]["line"],
    )

    #
    cbar = fig.colorbar(p, cax=cax)
cbar.ax.tick_params(labelsize=8)

ax.text(1.05, 250, "RH wrt ice", rotation=90, va="center", color="black", fontsize=16)
ax.text(
    1.02,
    290,
    "RH wrt \nliquid water",
    rotation=90,
    va="center",
    color="black",
    fontsize=16,
)
# cax.set_visible(False)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_ylim(305, 220)
ax.legend(loc=3)

ax.set_yticks([300, 280, 260, 240, 220])
ax.set_ylabel("Temperature / K")
ax.set_xlabel("Relative Humidity / 1")
sns.despine(offset={"bottom": 10})
sns.despine(ax=cax, bottom=True)
ax.axhline(273.15, color="gray", linestyle="--")
fig.tight_layout()
fig.savefig("/scratch/m/m301046/rh_ice_liq_hist_total.pdf", transparent=True)


# %%
fig, axes = plt.subplots(figsize=(17, 5.5), ncols=3, sharex=True, sharey=True)
cax = make_axes_locatable(axes[-1]).append_axes("right", size="5%", pad=0.55)

region = "total"
for ax, region in zip(axes, ["east", "west", "north"]):
    for name, label in [(f"lam-{region}", "ICON"), (f"beach-{region}", "Dropsondes")]:
        plthist = xr.concat(
            [
                (
                    (ice_histograms[name] / ice_histograms[name].sum("ice_rh_bin"))
                    .where(ice_histograms[name] > 0)
                    .sel(ta_bin=slice(None, 273))
                    .rolling(ta_bin=20, center=True, min_periods=3)
                    .sum()
                    .rolling(ice_rh_bin=7, center=True, min_periods=3)
                    .sum()
                ).rename({"ice_rh_bin": "rh_bin"}),
                (
                    (rh_histograms[name] / rh_histograms[name].sum("rh_bin"))
                    .where(rh_histograms[name] > 0)
                    .sel(ta_bin=slice(273, None))
                    .rolling(ta_bin=15, center=True, min_periods=3)
                    .sum()
                    .rolling(rh_bin=7, center=True, min_periods=3)
                    .sum()
                ),
            ],
            dim="ta_bin",
        )
        plthist = plthist / plthist.sum(("rh_bin"))

        if "beach" in name:
            fct = plthist.plot.contour
        else:
            fct = plthist.plot.contourf
        """
        """
        p = fct(
            y="ta_bin",
            ax=ax,
            levels=levels,
            **kwargs[name.split("-")[0]]["contour"],
        )

        (plthist * plthist.rh_bin).sum("rh_bin").sel(ta_bin=slice(None, 303)).plot(
            y="ta_bin",
            ax=ax,
            label=label,
            **kwargs[name.split("-")[0]]["line"],
        )

        #
        cbar = fig.colorbar(p, cax=cax)
cbar.ax.tick_params(labelsize=8)


# cax.set_visible(False)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_ylim(305, 220)
axes[0].legend(loc=3)

ax.set_yticks([300, 280, 260, 240, 220])
for ax in axes:
    ax.set_xlabel("Relative Humidity / 1")
    ax.set_ylabel("")
    ax.axhline(273.15, color="gray", linestyle="--")
    ax.text(
        1.05, 250, "RH wrt ice", rotation=90, va="center", color="black", fontsize=16
    )
    ax.text(
        1.02,
        290,
        "RH wrt \nliquid water",
        rotation=90,
        va="center",
        color="black",
        fontsize=16,
    )

axes[0].set_ylabel("Temperature / K")
sns.despine(offset={"bottom": 10})
sns.despine(ax=cax, bottom=True)
fig.tight_layout()
fig.savefig("/scratch/m/m301046/rh_ice_liq_hist_regions.pdf", transparent=True)


# %%
sns.set_context("talk", font_scale=0.9)
fig, ax = plt.subplots(figsize=(6, 5.5))
# cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.2)

levels = [0, 0.02, 0.04, 0.08, 0.1, 0.12]
region = "total"
for name, label in [(f"lam-{region}", "ICON"), (f"beach-{region}", "Dropsondes")]:
    plthist = (
        n2_hist[name]
        .rolling(ta_bin=5, center=True, min_periods=3)
        .sum()
        .where(n2_hist[name] > 0)
    )
    plthist = (plthist / plthist.sum("n2_bin")).sel(ta_bin=slice(260, 295))

    if "beach" in name:
        fct = plthist.plot.contour
    else:
        fct = plthist.plot.contourf
    """
    p=fct(
        y="ta_bin",
        ax=ax,
        levels=levels,
        **kwargs[name.split("-")[0]]["contour"],
    )
    """

    (plthist * plthist.n2_bin).sum("n2_bin").sel(ta_bin=slice(None, 303)).plot(
        y="ta_bin",
        ax=ax,
        label=label,
        **kwargs[name.split("-")[0]]["line"],
    )

    #
    # cbar=fig.colorbar(p, cax=cax)
# cbar.ax.tick_params(labelsize=8)
ax.plot(n2_pseudo, pseudo.T, color="black", linewidth=4, label="Pseudo-adiabat")

# gate
"""

plthist = n2_hist["gate"].rolling(ta_bin=5, center=True, min_periods=3).sum().where(n2_hist["gate"]>0)
plthist = (plthist / plthist.sum("n2_bin")).sel(
    ta_bin=slice(260, 295)
)
(plthist * plthist.n2_bin).sum("n2_bin").sel(ta_bin=slice(None, 303)).plot(
    y="ta_bin",
    ax=ax,
    label="GATE 1974",
    color="cornflowerblue",
    linewidth=4,
)
"""
# plt settings

ax.legend(loc=2)
# cax.set_visible(False)
ax.invert_yaxis()
# ax.set_xlim(0, 1)
# ax.set_ylim(305, 220)
ax.legend(loc=3, fontsize=12)

# ax.set_xticks([0.005, 0.01, 0.015])
ax.set_ylim([295, 260])
ax.set_ylabel("Temperature / K")
ax.set_xlabel("Brunt Väisälä Frequency ($N^2$) / s$^{-2}$")
# ax.set_xlim(0.003, 0.017)
ax.set_xlim(0.01, 0.014)

sns.despine(offset=10)
sns.despine(ax=cax, bottom=True, left=True)
ax.axhline(273.15, color="gray", linestyle="--")
fig.tight_layout()
# fig.savefig("/scratch/m/m301046/n2_hist_total.pdf", transparent=True)


# %%

# %%
sns.set_context("talk", font_scale=0.9)
fig, axes = plt.subplots(figsize=(17, 5.5), ncols=3, sharex=True, sharey=True)
cax = make_axes_locatable(axes[-1]).append_axes("right", size="5%", pad=0.55)

levels = [0, 0.02, 0.04, 0.08, 0.1, 0.12]
for ax, region in zip(axes, ["east", "west", "north"]):
    for name, label in [(f"lam-{region}", "ICON"), (f"beach-{region}", "Dropsondes")]:
        plthist = (
            n2_hist[name]
            .rolling(ta_bin=5, center=True, min_periods=3)
            .sum()
            .where(n2_hist[name] > 0)
        )
        plthist = (plthist / plthist.sum("n2_bin")).sel(ta_bin=slice(260, 295))

        if "beach" in name:
            fct = plthist.plot.contour
        else:
            fct = plthist.plot.contourf
        p = fct(
            y="ta_bin",
            ax=ax,
            levels=levels,
            **kwargs[name.split("-")[0]]["contour"],
        )

        (plthist * plthist.n2_bin).sum("n2_bin").sel(ta_bin=slice(None, 303)).plot(
            y="ta_bin",
            ax=ax,
            label=label,
            **kwargs[name.split("-")[0]]["line"],
        )

        #
        cbar = fig.colorbar(p, cax=cax)
cbar.ax.tick_params(labelsize=8)

# plt settings

axes[0].legend(loc=2)
ax.invert_yaxis()
for ax in axes:
    ax.set_xticks([0.005, 0.01, 0.015])
    ax.set_xlim(0.003, 0.017)
    ax.plot(n2_pseudo, pseudo.T, color="black", linewidth=4, label="Pseudo-adiabat")
    ax.set_xlabel("Brunt Väisälä Frequency ($N^2$) / s$^{-2}$")
    ax.axhline(273.15, color="gray", linestyle="--")

    ax.set_ylabel("")
axes[0].set_ylim([295, 260])
axes[0].set_ylabel("Temperature / K")
sns.despine(offset=10)
sns.despine(ax=cax, bottom=True, left=True)
fig.tight_layout()
# fig.savefig("/scratch/m/m301046/n2_plain_regions.pdf", transparent=True)
# %%

# %%

# %%
