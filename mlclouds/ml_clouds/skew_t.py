# %%
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cmo
from metpy.units import units
from metpy.plots import SkewT
from metpy.calc import dewpoint_from_specific_humidity
import metpy.calc as mpcalc
import sys

sys.path.append("../")
from myutils import open_datasets
from myutils.constants_and_values import ml_sondes


IPFS_GATEWAY = "https://ipfs.io"
cid = open_datasets.get_cid()
lvl3 = open_datasets.open_dropsondes(f"{cid}/dropsondes/Level_3/PERCUSION_Level_3.zarr")
# %%
mlcloud = lvl3.where(lvl3.sonde_id.isin(ml_sondes), drop=True)
lvl3_east = lvl3.where(lvl3.launch_lon > -40, drop=True)
lvl3_west = lvl3.where(lvl3.launch_lon < -40, drop=True)
mlcloud_east = mlcloud.where(mlcloud.launch_lon > -40, drop=True)
mlcloud_west = mlcloud.where(mlcloud.launch_lon < -40, drop=True)
no_mlcloud = lvl3.where(~lvl3.sonde_id.isin(ml_sondes), drop=True)
no_mlcloud_east = no_mlcloud.where(no_mlcloud.launch_lon > -40, drop=True)
no_mlcloud_west = no_mlcloud.where(no_mlcloud.launch_lon < -40, drop=True)
# %%


def plot_skew(skew, ds, c, label, wind=False, cin=False, xloc=1):
    p = ds.p.mean("sonde_id").sel(altitude=slice(0, 14000))
    p = p.where(p.values > np.insert(p[1:], -1, 0).values)
    p = p.interpolate_na("altitude", fill_value="extrapolate").values * units.Pa
    T = (
        ds.ta.mean("sonde_id")
        .interpolate_na("altitude", fill_value="extrapolate")
        .sel(altitude=slice(0, 14000))
        .values
        * units.kelvin
    )
    u = (
        ds.u.mean("sonde_id").sel(altitude=slice(0, 14000)).values
        * units.meter
        / units.second
    )
    v = (
        ds.v.mean("sonde_id").sel(altitude=slice(0, 14000)).values
        * units.meter
        / units.second
    )
    q = (
        ds.q.mean("sonde_id")
        .sel(altitude=slice(0, 14000))
        .interpolate_na("altitude", fill_value="extrapolate")
        .values
        * units.kilogram
        / units.kilogram
    )
    Td = dewpoint_from_specific_humidity(pressure=p, temperature=T, specific_humidity=q)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot((p), T, c=c, linewidth=2, label=label)
    skew.plot(p, Td, ":", c=c, linewidth=2)

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
    skew.plot(p, prof, "k", linewidth=1)

    # Shade areas of CAPE and CIN
    if cin:
        skew.shade_cin(p, T, prof, Td, alpha=0.5)
    skew.shade_cape(p, T, prof, color=c, alpha=0.5)
    if wind:
        skew.plot_barbs(
            p[::20],
            u[::20],
            v[::20],
            color=c,
            xloc=xloc,
            barb_increments={"half": 3, "full": 5, "flag": 50},
        )


# %%

cmap = cmo.tools.crop_by_percent(cmo.cm.tarn_r, 30, which="both")
west_color = cmap(0.2)
east_color = cmap(0.8)
# Change default to be better for skew-T
fig = plt.figure(figsize=(9, 6))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45, aspect=60)

for ds, c, label in [
    (lvl3_west, west_color, "west"),
    (lvl3_east, east_color, "east"),
]:
    plot_skew(skew, ds, c, label)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(
    t0=np.arange(248, 360, 10) * units.K, alpha=0.25, color="orangered"
)
skew.plot_moist_adiabats(
    t0=np.arange(248, 310, 10) * units.K,
    pressure=np.arange(1000, 100, -20) * units.hPa,
    alpha=0.25,
    color="tab:green",
)

skew.ax.axvline(
    270.3643761588233 - 273.15, color=west_color, linestyle="--", linewidth=2
)
skew.ax.axvline(
    267.8982991909455 - 273.15, color=east_color, linestyle="--", linewidth=2
)

skew.ax.set_ylim(None, 400)
skew.ax.set_xlim(5, 30)
skew.ax.yaxis.grid(False)
skew.ax.legend()
fig.savefig("../plots/skew_t_east_west.pdf", dpi=300, bbox_inches="tight")
# %%
