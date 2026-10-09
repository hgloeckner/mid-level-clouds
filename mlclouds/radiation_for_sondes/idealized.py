# %%

import xarray as xr
import numpy as np
import myutils.physics_helper as ph
import rad_helper as rh
import mlclouds.myutils.data_helper as datautils
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram
#
# %%

datapath = "/Users/helene/Documents/data/"
datapath = "/work/mh0066/m301046/data/"

beach = (
    xr.open_dataset(
        "ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q",
        engine="zarr",
    )
    .pipe(datautils.interpolate_gaps)
    .pipe(datautils.extrapolate_sfc)
)

expanded_beach = xr.open_dataset(datapath + "mlclouds/radiation_sondes.nc")

ct = xr.open_dataset(datapath + "wales/wales-ct.nc")
# %%


lowno_clouds = beach.where(
    (
        ct.sel(time=beach.launch_time.values, method="nearest").swap_dims(
            {"time": "sonde"}
        )["cloud-top"]
        < 4000
    )
    | (
        np.isnan(
            ct.sel(time=beach.launch_time.values, method="nearest").swap_dims(
                {"time": "sonde"}
            )["cloud-top"]
        )
    ),
    drop=True,
)
mid_clouds = beach.where(
    (
        ct.sel(time=beach.launch_time.values, method="nearest").swap_dims(
            {"time": "sonde"}
        )["cloud-top"]
        >= 4000
    )
    & (
        ct.sel(time=beach.launch_time.values, method="nearest").swap_dims(
            {"time": "sonde"}
        )["cloud-top"]
        < 8000
    ),
    drop=True,
)
mid_clouds = mid_clouds.sel(sonde=slice(3, None))
high_clouds = beach.where(
    (
        ct.sel(time=beach.launch_time.values, method="nearest").swap_dims(
            {"time": "sonde"}
        )["cloud-top"]
        >= 8000
    ),
    drop=True,
)
# %%

Px = beach.sel(altitude=slice(0, 50)).mean().p
P = np.arange(Px, 4000.0, -500)
Tsfc = beach.sel(altitude=slice(0, 50)).mean().ta
qsfc = beach.sel(altitude=slice(0, 50)).mean().q
plcl = mtf.plcl_bolton(T=Tsfc, P=Px, qt=qsfc)
zlcl = mtf.zlcl(plcl, T=Tsfc, P=Px, qt=qsfc, z=0)

adiabat = ph.make_sounding_from_adiabat(P, Tsfc.values, qsfc.values).rename(
    {"Trho": "ta", "P": "p"}
)
adiabat = xr.concat(
    [
        adiabat.sel(altitude=slice(None, 14000)),
        expanded_beach.mean("sonde").sel(altitude=slice(15000, None)),
    ],
    dim="altitude",
).reset_coords(["longitude"], drop=True)
# %%
es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)
qkwargs = {
    "CTH < 4 km": {
        "rhmid": 0.33,
        "rhlcl": 0.85,
        "rhtoa": None,
        "Ttop": None,
        "Tmin": 245,
        "rhtop": 0.7,
        # "zlcl": 440,
        "es": es,
        "rhpeak": 0.55,
        "tapeak": 273,
        "lowlim": 288,
        "talcl": 298,
        "highlim": 260,
    },
    "CTH 4-8 km": {
        "rhmid": 0.45,
        "Tmin": 245,
        "rhlcl": 0.8,
        "talcl": 298,
        "rhtoa": None,
        "Ttop": None,
        "rhtop": 0.7,
        # "zlcl": 510,
        "es": es,
        "rhpeak": 0.83,
        "tapeak": 273,
        "lowlim": 285,
        "highlim": 256,
    },
    "CTH > 8 km": {
        "rhmid": 0.58,  # 0.42
        "rhlcl": 0.87,  # 0.9
        "Ttop": 220,
        "rhtop": 0.9,
        "rhtoa": 0.75,  # 0.35
        "Tmin": 260,  # 250
        "zlcl": 543,
        "es": es,
        "talcl": 298,
        "rhpeak": 0.73,
        "tapeak": 273,
        "lowlim": 284,
        "highlim": 263,  # 262
    },
}


def get_qshapes(dst):
    qshapes = []
    rhshapes = []
    for shape in qkwargs.keys():
        qc, rhc = ph.cshape_humidity(dst, **qkwargs[shape])
        qc = xr.concat(
            [
                qc.sel(altitude=slice(None, 14000)),
                expanded_beach.mean("sonde").sel(altitude=slice(15000, None)).q,
            ],
            dim="altitude",
        ).reset_coords(["longitude"], drop=True)
        qe, rhe = ph.eshape_humidity(dst, **qkwargs[shape])
        qe = xr.concat(
            [
                qe.sel(altitude=slice(None, 14000)),
                expanded_beach.mean("sonde").sel(altitude=slice(15000, None)).q,
            ],
            dim="altitude",
        ).reset_coords(["longitude"], drop=True)
        qshapes.append(qc)
        rhshapes.append(rhc)
        qshapes.append(qe)
        rhshapes.append(rhe)
    return qshapes, rhshapes


def get_qreal(dst):
    qreals = []
    coords = []
    tbins = np.linspace(200, 310, 200)
    for ds in [
        lowno_clouds,
        mid_clouds,
        high_clouds,
        beach,
    ]:  # lowno_clouds, , high_clouds, beach
        ds = ds.sel(altitude=slice(None, 14500))
        rhreal = (
            ds.groupby_bins(ds.ta, bins=tbins, labels=(tbins[:-1] + tbins[1:]) / 2)
            .mean()
            .drop_vars("ta")
            .rename({"ta_bins": "ta"})
        )
        rhreal = (
            rhreal.interp(ta=dst.ta.values)
            .assign(altitude=dst.swap_dims({"altitude": "ta"}).altitude)
            .swap_dims({"ta": "altitude"})
        )

        qreal = mtf.relative_humidity_to_specific_humidity(
            rhreal.rh, dst.p, dst.ta, es=svp.liq_hardy
        )
        print(qreal)
        print(
            expanded_beach.mean("sonde")
            .reset_coords()
            .sel(altitude=slice(15000, None))
            .q
        )
        qreals.append(
            xr.concat(
                [
                    qreal.sel(altitude=slice(None, 14000)).reset_coords(drop=True),
                    expanded_beach.mean("sonde")
                    .reset_coords()
                    .sel(altitude=slice(15000, None))
                    .q,
                ],
                dim="altitude",
            )
        )
        coords.append(
            dict(
                launch_time=ds.launch_time.mean().values,
                launch_lat=ds.launch_lat.mean().values,
                launch_lon=ds.launch_lon.mean().values,
            )
        )

    return qreals, coords


def get_different_shapes(dst):
    qshapes, _ = get_qshapes(dst)
    qreals, coords = get_qreal(dst)

    dst = dst.assign(
        q=(
            ("shape", "altitude"),
            xr.concat(qshapes + qreals, dim="shape")
            .bfill(dim="altitude")
            .ffill(dim="altitude")
            .values,
        ),
        rh=(
            ("shape", "altitude"),
            xr.concat(
                [
                    mtf.specific_humidity_to_relative_humidity(q, dst.p, dst.ta, es=es)
                    for q in qshapes + qreals
                ],
                dim="shape",
            ).values,
        ),
        o3=(
            ("altitude"),
            expanded_beach.mean("sonde").o3.interp(altitude=dst.altitude).values,
        ),
        shape=(
            "shape",
            [
                "clow",
                "elow",
                "cmid",
                "emid",
                "chigh",
                "ehigh",
                "reallow",
                "realmid",
                "realhigh",
                "realall",
            ],
        ),
    )
    return dst


adhum = get_different_shapes(adiabat)

import seaborn as sns

colors = sns.color_palette("Paired")
altslice = slice(0, 15500)

fig, ax = plt.subplots(figsize=(6, 4))

for idx, (shape, ls) in enumerate(
    [
        ("c", ":"),
        ("real", "-"),
        ("e", "-"),
    ]
):
    for ct, color in [("low", 0), ("mid", 2), ("high", 4)]:
        ax.plot(
            adhum.rh.sel(shape=shape + ct, altitude=altslice),
            adhum.ta.sel(altitude=altslice),
            label=shape + ct,
            color=colors[color + idx % 2],
            ls=ls,
        )

ax.legend()
ax.invert_yaxis()
# plt.ylim(0, 15000)

# %%


# %%
# xr.broadcast(adiabat.rename({"shape":"sonde"}))[0].transpose("sonde", "altitude").to_netcdf(datapath + "mlclouds/realt_profiles.nc")
# idealized_profiles.nc for moist adiabatic T


# %% real T idealized profiles
P = np.arange(Px, 4000.0, -500)[::-1]
pres = np.arange(Px + 250, 3750.0, -500)[::-1]

beach_on_grid = (
    beach.groupby_bins(beach.p, pres, labels=P)
    .mean()
    .drop_vars("p")
    .rename({"p_bins": "p"})
)
beach_on_grid = (
    beach_on_grid.assign(
        altitude=adiabat.swap_dims({"altitude": "p"}).interp(p=beach_on_grid.p).altitude
    )
    .swap_dims({"p": "altitude"})
    .sortby("altitude")
)
# %%

realt = xr.concat(
    [
        beach_on_grid.sel(altitude=slice(None, 14000)).reset_coords(["p"]),
        expanded_beach.mean("sonde").sel(altitude=slice(15000, None)),
    ],
    dim="altitude",
).reset_coords(["longitude"], drop=True)

# %%
realhum = get_different_shapes(realt)

import seaborn as sns

colors = sns.color_palette("Paired")
altslice = slice(0, 15500)

fig, ax = plt.subplots(figsize=(6, 4))

for idx, (shape, ls) in enumerate(
    [
        ("c", ":"),
        ("real", "-"),
        ("e", "-"),
    ]
):
    for ct, color in [("low", 0), ("mid", 2), ("high", 4)]:
        ax.plot(
            realhum.rh.sel(shape=shape + ct, altitude=altslice),
            realhum.ta.sel(altitude=altslice),
            label=shape + ct,
            color=colors[color + idx % 2],
            ls=ls,
        )
        ax.plot(
            adhum.rh.sel(shape=shape + ct, altitude=altslice),
            adhum.ta.sel(altitude=altslice),
            label=shape + ct,
            color=colors[color + idx % 2],
            ls=ls,
        )

# ax.legend()
ax.invert_yaxis()
# plt.ylim(0, 15000)
# %%
humdata = {"adiabat": adhum, "real": realhum}

for key, ds in humdata.items():
    humdata[key] = ds.assign(
        q=xr.where(
            ds.altitude >= 25000, 4e-8, xr.where(ds.altitude < 20000, ds.q, np.nan)
        )
        .chunk({"altitude": -1})
        .interpolate_na(dim="altitude")
        .T
    )
    humdata[key] = ds.assign(
        q=xr.where(
            ds.altitude < 500, ds.sel(altitude=500, method="nearest").q.values, ds.q
        )
    )
    xr.broadcast(humdata[key].rename({"shape": "sonde"}))[0].transpose(
        "sonde", "altitude"
    ).to_netcdf(datapath + f"mlclouds/idealized_{key}_profiles.nc")


# %%
for ds in humdata.values():
    # ds = ds.swap_dims({"altitude":"ta"})
    ds.q.sel(shape="realall").plot(y="altitude")

# expanded_beach.mean("sonde").q.plot.line(y="altitude", color="k", ls="--")
beach.q.mean("sonde").plot.line(y="altitude", color="k", ls=":")
# plt.xlim(0.0, 0.00002)
plt.ylim(0, 15000)
# %%
