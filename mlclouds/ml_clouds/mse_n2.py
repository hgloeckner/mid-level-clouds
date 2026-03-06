# %%
import seaborn as sns
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import scipy
import cmocean as cmo
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import sys

sys.path.append("../")
import lam_orcestra.helper as help
import myutils.data_helper as dh
from mpl_toolkits.axes_grid1 import make_axes_locatable
import eurec4a

colors = {
    "beach": "teal",
    "North": "#FF7982",
    "East": "#960018",
    "West": "#0085db",
    "joanne": "#FFA500",
    "gate": "green",
}
# %%
cat = eurec4a.get_intake_catalog()
joanne = cat.dropsondes.JOANNE.level3.to_dask().rename(
    {
        "alt": "altitude",
    }
)


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


# %%
"""
lam_sondes = xr.open_dataset(
    "/scratch/m/m301046/lam_sondes_z.zarr",
    engine="zarr",
)
ifs = xr.open_dataset(
    "/work/mh0492/m301067/orcestra/results/timeseries/ifs_interpolated_on_dropsondes_profiles_2nd-days.nc"
)
"""

cids = get_cids()
beach = (
    open_dropsondes(cids["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
    .isel(sonde=slice(2, None))
    .reset_coords(["launch_lat", "launch_lon"])
).swap_dims({"sonde": "sonde_id"})
joanne = joanne.pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc)
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
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
).swap_dims({"sonde": "sonde_id"})
# %% create lam dataset
"""
lam_ds = lam_sondes.where(
    lam_sondes.sonde_id.isin(beach.sonde_id.values),
    drop=True,
)
altdim = "z"

lam_subset = lam_ds.sortby(altdim, ascending=False).reset_coords(
    ["launch_lat", "launch_lon"]
).rename(
    {
        "qv":"q",
        "pfull":"p",
        "z":"altitude",
    }
)
lam_subset = lam_subset.assign(
    rh=mtf.specific_humidity_to_relative_humidity(
        q=lam_subset.q,
        p=lam_subset.p,
        T=lam_subset.ta,
        es=svp.liq_wagner_pruss,
    ),
    theta=mtf.theta(lam_subset.ta, lam_subset.p),
).sel(z=slice(15000, None))

ifs = ifs.assign(
    rh=mtf.specific_humidity_to_relative_humidity(
        q=ifs.q,
        p=ifs.pressure,
        T=ifs.t,
        es=svp.liq_wagner_pruss,
    ),
    theta=mtf.theta(ifs.t, ifs.pressure),
).rename({"t": "ta","pressure":"p", "height": "altitude", "sonde_lat": "launch_lat"})
"""
# %%
data = {}
for ds, name in zip(
    [beach, gate, joanne], ["beach", "gate", "joanne"]
):  # lam_subset, ifs]:
    ds = ds.assign(
        n2=help.apply_brunt_vaisala_frequency(ds, altdim="altitude", q="q"),
        ice_rh=mtf.specific_humidity_to_relative_humidity(
            q=ds.q,
            p=ds.p,
            T=ds.ta,
            es=svp.ice_wagner_etal,
        ),
        mse=help.apply_mse(ds, altdim="altitude", q="q"),
    )
    midn2 = ds.n2.rolling(altitude=100).mean().sel(altitude=slice(2500, 8000))
    dmax = midn2.sel(altitude=slice(4000, 8000)).dropna(dim="sonde_id", how="all")
    max_alts = dmax.isel(altitude=dmax.argmax(dim="altitude")).altitude
    dmin = midn2.where(midn2.altitude < max_alts).dropna(dim="sonde_id", how="all")
    min_alts = dmin.isel(altitude=dmin.argmin(dim="altitude")).altitude

    ds = ds.assign(
        n2maxalt=dmax.isel(altitude=dmax.argmax(dim="altitude")).altitude,
        n2minalt=dmin.isel(altitude=dmin.argmin(dim="altitude")).altitude,
        n2max=dmax.isel(altitude=dmax.argmax(dim="altitude")),
        n2min=dmin.isel(altitude=dmin.argmin(dim="altitude")),
        n2diff=dmax.isel(altitude=dmax.argmax(dim="altitude"))
        - dmin.isel(altitude=dmin.argmin(dim="altitude")),
        mse_low=ds.mse.sel(altitude=slice(0, 300)).mean(dim="altitude"),
    )
    data[name] = ds

# %%


# %%
x = np.linspace(0, 0.013)
sns.set_palette("tab10")
fig, axes = plt.subplots(
    figsize=(6, 6), ncols=2, nrows=2, width_ratios=[0.8, 0.2], height_ratios=[0.2, 0.8]
)
for region, name in [(dh.east, "East"), (dh.west, "West"), (dh.north, "North")]:
    ds = dh.sel_sub_domain(data["beach"], region, item_var="sonde_id")

    sns.histplot(
        ds.n2diff,
        ax=axes[0, 0],
        stat="density",
        binrange=[0, 0.013],
        bins=30,
        kde=True,
        color=colors[name],
        element="step",
        alpha=0.1,
        label=name,
    )
    sns.histplot(
        ds.mse_low.reset_coords(drop=True).to_dataframe(),
        y="mse_low",
        ax=axes[1, 1],
        stat="density",
        binrange=[330000, 385000],
        bins=30,
        kde=True,
        color=colors[name],
        element="step",
        alpha=0.1,
        label=name,
    )

    axes[1, 0].scatter(
        ds.n2diff, ds.mse_low, s=10, label=name, alpha=0.2, c=colors[name]
    )
    ds = ds.dropna(dim="sonde_id", subset=["n2diff", "mse_low"])
    regres = scipy.stats.linregress(
        ds.n2diff.values,
        ds.mse_low.values,
    )
    axes[1, 0].plot(x, regres.slope * x + regres.intercept, c=colors[name], alpha=0.5)
    print("p-val", regres.pvalue, "r-value", regres.rvalue)
for name in ["gate", "joanne"]:
    axes[1, 0].scatter(
        data[name].n2diff,
        data[name].mse_low,
        s=10,
        label=name,
        c=colors[name],
        alpha=0.2,
    )
    ds = data[name].dropna(dim="sonde_id", subset=["n2diff", "mse_low"])
    regres = scipy.stats.linregress(
        ds.n2diff.values,
        ds.mse_low.values,
    )
    axes[1, 0].plot(x, regres.slope * x + regres.intercept, c=colors[name], alpha=0.5)
    print("p-val", regres.pvalue, "r-value", regres.rvalue)
    sns.histplot(
        ds.n2diff,
        ax=axes[0, 0],
        stat="density",
        binrange=[0, 0.013],
        bins=30,
        kde=True,
        color=colors[name],
        element="step",
        alpha=0.1,
        label=name,
    )

    sns.histplot(
        ds.mse_low.reset_coords(drop=True).to_dataframe(),
        y="mse_low",
        ax=axes[1, 1],
        stat="density",
        binrange=[330000, 385000],
        bins=30,
        kde=True,
        color=colors[name],
        element="step",
        alpha=0.1,
        label=name,
    )

axes[0, 0].legend(fontsize=8)
axes[1, 1].set_ylabel("")
axes[1, 0].set_xlabel(r"$\Delta N^2$")
axes[1, 0].set_ylabel("MSE mean (0m - 300m)")
# ax.set_ylim(350000, 375000)
axes[0, 1].set_visible(False)


sns.despine(offset=5)
# %%


# %%
sns.set_palette("Paired")
fig, ax = plt.subplots()
for dsname in ["beach", "joanne", "gate"]:
    q = (
        data[dsname]
        .mse.sel(altitude=slice(0, 300))
        .mean("altitude")
        .quantile([0.1, 0.9])
    )

    data[dsname].where(
        data[dsname].mse.sel(altitude=slice(0, 300)).mean("altitude") > q[1].values,
        drop=True,
    ).n2.mean("sonde_id").rolling(altitude=50).mean().plot(
        y="altitude", label=f"{dsname} 10% high MSE < 300m"
    )
    data[dsname].where(
        data[dsname].mse.sel(altitude=slice(0, 300)).mean("altitude") < q[0].values,
        drop=True,
    ).n2.mean("sonde_id").rolling(altitude=50).mean().plot(
        y="altitude", label="       10% low MSE < 300m"
    )
ax.set_ylim(0, 7000)
ax.set_xlim(0.005, 0.018)
ax.legend(fontsize=8)
sns.despine()
# %%
dsname = "beach"
sns.set_palette("Paired")
fig, ax = plt.subplots()
for region, name in [(dh.east, "East"), (dh.west, "West"), (dh.north, "North")]:
    ds = dh.sel_sub_domain(data[dsname], region, item_var="sonde_id")
    q = ds.mse.sel(altitude=slice(0, 300)).mean("altitude").quantile([0.1, 0.9])

    ds.where(
        ds.mse.sel(altitude=slice(0, 300)).mean("altitude") > q[1].values, drop=True
    ).n2.mean("sonde_id").rolling(altitude=50).mean().plot(
        y="altitude", label=f"{name} 10% high MSE < 300m"
    )
    ds.where(
        ds.mse.sel(altitude=slice(0, 300)).mean("altitude") < q[0].values, drop=True
    ).n2.mean("sonde_id").rolling(altitude=50).mean().plot(
        y="altitude", label="       10% low MSE < 300m"
    )
ax.set_ylim(0, 7000)
ax.set_xlim(0.005, 0.018)
ax.legend(fontsize=8)
sns.despine()
