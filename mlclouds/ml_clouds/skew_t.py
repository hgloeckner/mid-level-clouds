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
from myutils import open_datasets as od
from myutils import data_helper as dh


IPFS_GATEWAY = "https://ipfs.io"
cids = od.get_cids()
lvl3 = od.open_dropsondes(cids["dropsondes"])
# %%
east = dh.sel_sub_domain(lvl3, dh.east)
west = dh.sel_sub_domain(lvl3, dh.west)
# %%


def plot_skew(skew, ds, c, label, wind=False, cin=False, dew=True, xloc=1):
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
    if dew:
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
        skew.plot_barbs(
            p[::20],
            u[::20],
            v[::20],
            color=c,
            xloc=xloc,
            barb_increments={"half": 3, "full": 5, "flag": 50},
        )


# %%

from myutils import physics_helper as ph
from moist_thermodynamics import functions as mtf

Psfc = lvl3.p.mean(dim="sonde").sel(altitude=0).values
P = np.arange(Psfc, 4000.0, -500)
sfcT = lvl3.ta.mean(dim="sonde").sel(altitude=0).values
qsfc = lvl3.q.mean(dim="sonde").sel(altitude=0).values  # 9182267570514704

pseudo = (
    ph.make_sounding_from_adiabat(P, sfcT, qsfc, thx=mtf.theta_e_bolton, Tmin=195)
    .rename({"P": "p", "T": "ta", "q": "q"})
    .expand_dims({"sonde_id": [0]})
)


# %%
east_color = cmo.cm.phase(0.25)
west_color = cmo.cm.phase(0.75)

# Change default to be better for skew-T
fig = plt.figure(figsize=(9, 6))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45, aspect=60)

for ds, c, label in [
    (west.swap_dims({"sonde": "sonde_id"}), west_color, "west"),
    # (east.swap_dims({"sonde":"sonde_id"}), east_color, "east"),
]:
    plot_skew(skew, ds, c, label="unstable", dew=False)
    plot_skew(
        skew,
        lvl3.isel(sonde=2)
        .expand_dims({"sonde_id": [0]})[["p", "ta", "q"]]
        .rolling(altitude=10)
        .mean(),
        east_color,
        label="stable",
        dew=False,
    )

# plot_skew(skew, pseudo, "k", label="pseudo-adiabat", wind=False, dew=False)

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
"""
skew.ax.axvline(
    270.3643761588233 - 273.15, color=west_color, linestyle="--", linewidth=2
)
skew.ax.axvline(
    267.8982991909455 - 273.15, color=east_color, linestyle="--", linewidth=2
)
"""
skew.ax.set_ylim(None, 400)
skew.ax.set_xlim(5, 30)
skew.ax.yaxis.grid(False)
skew.ax.legend()
# fig.savefig("../plots/skew_t_east_west.pdf", dpi=300, bbox_inches="tight")
# %%
