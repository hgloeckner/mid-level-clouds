#!/usr/bin/env python3
#SBATCH --account=mh0066
#SBATCH --partition=compute
#SBATCH --time=01:00:00

#%%
import numpy as np
import xarray as xr
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)
import pyarts
from pyarts.arts import convert
import FluxSimulator as fsm
import myutils.physics_helper as ph
import time
import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")

from radiation_for_sondes import angles
from radiation_for_sondes import rad_helper as rh


sonde_data = xr.open_dataset(
    "/work/mh0066/m301046/data/mlclouds/idealized_profiles.nc"
    
)

mean_day = np.datetime64("2024-09-01")
mean_lat = 9
mean_lon = -40

sonde_data = sonde_data.assign_coords({"hour_of_day": ("hour_of_day", np.arange(0, 24))})
#%% RRTMG

rrtmg_data = sonde_data.assign(
    mu0=xr.apply_ufunc(
        angles.cos_zenith_angle,
        mean_day + sonde_data.hour_of_day.astype("timedelta64[h]"),
        mean_lat,
        mean_lon,
        vectorize=True,
    )
)

rrtmg_atm = rh.make_rrtmg_atm(
    rrtmg_data
    )

gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
optical_props = gas_optics_lw.compute(rrtmg_atm, add_to_input=False)
optical_props = optical_props.assign(surface_emissivity=1 - rh.lw_reflectivity)

lw_fluxes = optical_props.rte.solve(add_to_input=False)
optical_props_sw = gas_optics_sw.compute(rrtmg_atm, add_to_input=False)

optical_props_sw["surface_albedo"] = rh.sw_reflectivity

sw_fluxes = []
for mu0 in rrtmg_data.mu0.values:
    optical_props_sw = optical_props_sw.assign(mu0=("column", np.repeat(mu0, len(rrtmg_atm.column))))
    sw_fluxes.append(optical_props_sw.rte.solve(add_to_input=False).assign(mu0=[mu0]))


ds = xr.merge([lw_fluxes, xr.concat(sw_fluxes, dim="mu0"), rrtmg_atm]).rename({"column":"sonde"}).assign(
    altitude = ("level", rrtmg_data.altitude.values),
)
ds.attrs = {}
ds = ds.swap_dims({"level": "altitude"}).assign(
    sonde = ("sonde", rrtmg_data.sonde.values),
    hour_of_day = ("mu0", rrtmg_data.hour_of_day.values),
)
ds.to_zarr(
    #"rrtmgp_sonde_fluxes.zarr", 
    "/work/mh0066/m301046/data/idealized_rrtmg_fluxes.zarr",
    encoding=rh.get_encoding(ds),
    mode="w",
    zarr_format=2
)

