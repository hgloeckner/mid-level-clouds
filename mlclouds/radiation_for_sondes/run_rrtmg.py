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

import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")

from radiation_for_sondes import angles
from radiation_for_sondes import rad_helper as rh


sonde_data = xr.open_dataset(
    #"/scratch/m/m301046/sondes_for_radiation.nc"
    "/work/mh0066/m301046/data/idealized_profiles.nc"
    
)
sonde_data = sonde_data.assign(
    mu0=xr.apply_ufunc(
        angles.cos_zenith_angle,
        sonde_data.launch_time,
        sonde_data.launch_lat,
        sonde_data.launch_lon,
        vectorize=True,
    )
)

rrtmg_atm = rh.make_rrtmg_atm(
    sonde_data
    )

gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
optical_props = gas_optics_lw.compute(rrtmg_atm, add_to_input=False)
optical_props = optical_props.assign(surface_emissivity=1 - rh.lw_reflectivity)

lw_fluxes = optical_props.rte.solve(add_to_input=False)
optical_props_sw = gas_optics_sw.compute(rrtmg_atm, add_to_input=False)

optical_props_sw["surface_albedo"] = rh.sw_reflectivity
optical_props_sw = optical_props_sw.assign(mu0=("column", sonde_data.mu0.values))
sw_fluxes = optical_props_sw.rte.solve(add_to_input=False)

#%%

ds = xr.merge([lw_fluxes, sw_fluxes, rrtmg_atm]).rename({"column":"sonde"}).assign(
    altitude = ("level", sonde_data.altitude.values),
)
#%%
ds.attrs = {}
#%%
ds.to_zarr(
    #"rrtmgp_sonde_fluxes.zarr", 
    "/work/mh0066/m301046/data/idealized_rrtmg_fluxes.zarr",
    encoding=rh.get_encoding(ds),
    mode="w",
    zarr_format=2
)
