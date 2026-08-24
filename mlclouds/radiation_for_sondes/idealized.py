#%%

import xarray as xr
import numpy as np
import myutils.physics_helper as ph
import rad_helper as rh
import mlclouds.myutils.data_helper as datautils
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram

#%%

beach = (xr.open_dataset("ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr")
         .pipe(datautils.interpolate_gaps).pipe(datautils.extrapolate_sfc)
         
         )

expanded_beach = xr.open_dataset(
    "/scratch/m/m301046/radiation_sondes.nc"
)

ct = xr.open_dataset(
    "/work/mh0066/m301046/data/wales-ct.nc"
)
#%%

beach_ta = histogram(

)


lowno_clouds = beach.where(
    (ct.sel(time=beach.launch_time.values, method="nearest").swap_dims({"time":"sonde"})['cloud-top'] < 4000) | (np.isnan(ct.sel(time=beach.launch_time.values, method="nearest").swap_dims({"time":"sonde"})['cloud-top'])), 
    drop=True)
mid_clouds = beach.where(
    (ct.sel(time=beach.launch_time.values, method="nearest").swap_dims({"time":"sonde"})['cloud-top'] >= 4000) & (ct.sel(time=beach.launch_time.values, method="nearest").swap_dims({"time":"sonde"})['cloud-top'] < 8000), 
    drop=True)
high_clouds = beach.where(
    (ct.sel(time=beach.launch_time.values, method="nearest").swap_dims({"time":"sonde"})['cloud-top'] >= 8000), 
    drop=True)
#%%

Px = beach.sel(altitude=slice(0, 50)).mean().p
P = np.arange(Px, 4000.0, -500)
Tsfc = beach.sel(altitude=slice(0, 50)).mean().ta
qsfc = beach.sel(altitude=slice(0, 50)).mean().q
plcl = mtf.plcl_bolton(T=Tsfc, P=Px, qt=qsfc)
zlcl = mtf.zlcl(plcl, T=Tsfc, P=Px, qt=qsfc, z=0)

adiabat = ph.make_sounding_from_adiabat(
    P, Tsfc.values, qsfc.values
).rename({"Trho":"ta", "P":"p"})
adiabat = xr.concat(
    [adiabat.sel(altitude=slice(None, 15000)),
     expanded_beach.mean("sonde").sel(altitude=slice(17000, None)),
     ],
     dim="altitude"
).reset_coords(["longitude"], drop=True)
#%%

es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)
qkwargs = {
    "CTH < 4 km": {
        "rhmid": 0.5,
        "rhlcl": 0.9,
        "rhtoa": 0.9,
        "Tmin": 280,
        "zlcl": 440,
        "es": es,
        "factor": 0.4,
        "lowlim": 286,
        "highlim": 260,
    },
    "CTH 4-8 km": {
        "rhmid": 0.4,
        "rhlcl": 0.87,
        "rhtoa": 0.8,
        "Tmin": 250,
        "zlcl": 510,
        "es": es,
        "factor": 0.65,
        "lowlim": 287,
        "highlim": 255,
    },
    "CTH > 8 km": {
        "rhmid": 0.55,  # 0.42
        "rhlcl": 0.88,  # 0.9
        "rhtoa": 1.,  # 0.35
        "Tmin": 265,  # 250
        "zlcl": 543,
        "es": es,
        "factor": 0.52,  # 0.52
        "lowlim": 286,
        "highlim": 260,  # 262
    },
}
qshapes = []
for shape in qkwargs.keys():
    qshapes.append(ph.cshape_humidity(
        adiabat, **qkwargs[shape]
    ))
    qshapes.append(ph.eshape_humidity(
        adiabat, **qkwargs[shape]
    ))
qreals= []
coords = []
for ds in [lowno_clouds, mid_clouds, high_clouds, beach]:
    qreals.append(
        mtf.relative_humidity_to_specific_humidity(
            ds.mean("sonde").rh.interp(altitude=adiabat.altitude),
            adiabat.p,
            adiabat.ta,
            es = svp.liq_hardy
        )
    )
    coords.append(xr.Dataset({
        "launch_time": ds.launch_time.mean("sonde"),
        "launch_lat": ds.launch_lat.mean("sonde"),
        "launch_lon": ds.launch_lon.mean("sonde")
    }))

adiabat = adiabat.assign(
    q = (("shape", "altitude"), xr.concat(
        qshapes + qreals, dim="shape"
    ).bfill(dim="altitude").ffill(dim="altitude").values),
    rh = (("shape", "altitude"), xr.concat(
        [mtf.specific_humidity_to_relative_humidity(q, adiabat.p, adiabat.ta, es=es) for q in qshapes + qreals], 
        dim="shape"
    ).values),
    o3 = (("altitude"), expanded_beach.mean("sonde").o3.interp(altitude=adiabat.altitude).values),
    shape = ("shape", ["clow", "elow","cmid", "emid", "chigh","ehigh", "reallow", "realmid", "realhigh", "realall"])
)
adiabat = adiabat.assign_coords(
    {   

        coord: (("shape"), [coords[0][coord].values]*2 + [coords[1][coord].values]*2 + [coords[2][coord].values]*2 + [coords[i][coord].values for i in range(3)]+ [coords[3][coord].values])
        for coord in coords[0].variables.keys()
    }
)

import seaborn as sns
colors = sns.color_palette("Paired")
altslice=slice(0, 15500)

fig, ax = plt.subplots(figsize=(6, 4))

for idx, (shape, ls) in enumerate([("c", ":"), ("real", "-"), ("e", "-"), ]):
    for ct, color in [("low", 0), ("mid", 2), ("high", 4)]:
        ax.plot(
            adiabat.rh.sel(shape=shape+ct, altitude=altslice),
            adiabat.ta.sel(altitude=altslice),
            label=shape+ct,
            color=colors[color + idx % 2],
            ls=ls
        )

ax.legend()
ax.invert_yaxis()
#plt.ylim(0, 15000)

#%%

adiabat = adiabat.assign(
q = xr.where(adiabat.altitude >= 25000, 4e-8 , xr.where(adiabat.altitude < 20000, adiabat.q, np.nan)).chunk({"altitude":-1}).interpolate_na(dim="altitude").T

)


#%%
xr.broadcast(adiabat.rename({"shape":"sonde"}))[0].transpose("sonde", "altitude").to_netcdf("/work/mh0066/m301046/data/idealized_profiles.nc")

# %%

from radiation_for_sondes import rad_helper as rh

rh.make_rrtmg_atm(
    xr.broadcast(adiabat.rename({"shape":"sonde"}))[0].transpose("sonde", "altitude")
)