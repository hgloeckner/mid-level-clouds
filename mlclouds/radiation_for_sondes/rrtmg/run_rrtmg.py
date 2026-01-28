#!/usr/bin/env python3
# SBATCH --account=mh0066
# SBATCH --partition=compute
# SBATCH --time=04:00:00

# %%
import xarray as xr
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles
import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")
from radiation_for_sondes.rrtmg import angles
import numpy as np


def make_atmosphere(p, T, h2o_vmr, o3, T_s=None):
    """Create a pyRTE-RRTMG atmosphere from pressure, temperature and humidity arrays."""
    if np.any(p[..., 0] < p[..., -1]):
        raise ValueError("Arrays need to be passed in ascending order")

    if T_s is None:
        T_s = T[..., 0]

    atmosphere = xr.Dataset(
        data_vars={
            "pres_level": (("column", "level"), p),
            "temp_level": (("column", "level"), T),
            "pres_layer": (("column", "layer"), 0.5 * (p[..., 1:] + p[..., :-1])),
            "temp_layer": (("column", "layer"), 0.5 * (T[..., 1:] + T[..., :-1])),
            "surface_temperature": (("column",), T_s),
            "h2o": (("column", "layer"), 0.5 * (h2o_vmr[..., 1:] + h2o_vmr[..., :-1])),
            "o3": (("layer"), 0.5 * (o3[..., 1:] + o3[..., :-1])),
            "co2": 422e-6,
            "ch4": 1650e-9,
            "n2o": 306e-9,
            "n2": 0.7808,
            "o2": 0.2095,
            "co": 0.0,
        },
    )

    return atmosphere


# %%
ds = xr.open_dataset(
    "/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc"
).swap_dims({"sonde": "sonde_id"})
lvl3 = xr.open_dataset(
    "ipfs://bafybeiesyutuduzqwvu4ydn7ktihjljicywxeth6wtgd5zi4ynxzqngx4m", engine="zarr"
).swap_dims({"sonde": "sonde_id"})
# ds = ds.pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc)
# ds = ds.isel(  # pick every 10th sonde
#    altitude=slice(0, 1100),  # skip upper-most layers due to NaN values
# ).dropna(dim="sonde", how="any", subset=["ta", "q", "p"])

ds = ds.assign(
    launch_lat=lvl3.launch_lat.sel(sonde_id=ds.sonde_id),
    launch_lon=lvl3.launch_lon.sel(sonde_id=ds.sonde_id),
    launch_time=lvl3.launch_time.sel(sonde_id=ds.sonde_id),
)  # .isel(sonde_id=slice(45, 50))

ds = ds.swap_dims({"sonde_id": "sonde"}).assign(
    mu0=xr.apply_ufunc(
        angles.cos_zenith_angle,
        ds.launch_time,
        ds.launch_lat,
        ds.launch_lon,
        vectorize=True,
    )
)

atmosphere = make_atmosphere(ds.p.values, ds.t.values, ds.H2O.values, ds.O3.values)
atmosphere
# %%
gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
optical_props = gas_optics_lw.compute(atmosphere, add_to_input=False)

optical_props = optical_props.assign(surface_emissivity=0.98)
# %%
clr_fluxes = optical_props.rte.solve(add_to_input=False)
optical_props_sw = gas_optics_sw.compute(atmosphere, add_to_input=False)

optical_props_sw["surface_albedo"] = 0.06
optical_props_sw = optical_props_sw.assign(mu0=("column", ds.mu0.values))
sw_fluxes = optical_props_sw.rte.solve(add_to_input=False)

# %%
atmosphere = atmosphere.assign_coords(
    {
        "sonde_id": ("column", ds.sonde_id.values),
        "launch_time": ("column", ds.launch_time.values),
        "launch_lat": ("column", ds.launch_lat.values),
        "launch_lon": ("column", ds.launch_lon.values),
    }
)
xr.merge([clr_fluxes, sw_fluxes, atmosphere, optical_props_sw[["mu0"]]]).to_zarr(
    "/scratch/m/m301046/rrtmgp_sonde_fluxes.zarr", mode="w", zarr_format=2
)
