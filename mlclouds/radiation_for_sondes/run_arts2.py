#!/usr/bin/env python3
#SBATCH --account=mh0066
#SBATCH --partition=compute
#SBATCH --array=0-112%10
#SBATCH --time=01:00:00


import os
import dask
import numpy as np
import xarray as xr
import pyarts
from pyarts.arts import convert
import FluxSimulator as fsm
import time
import sys
import argparse

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")

from radiation_for_sondes import rad_helper as rh
import myutils.physics_helper as ph


sondes_for_rad = "/work/mh0066/m301046/data/idealized_profiles.nc" #"/scratch/m/m301046/radiation_sondes.nc"
min_wavelength_sw = 3e-7  # [m]
max_wavelength_sw = 5e-6  # [m]
n_freq_sw = 200
surface_reflectivity_sw = rh.sw_reflectivity
surface_reflectivity_lw = rh.lw_reflectivity
surface_altitude= 0.0  # [m]

wvl = np.linspace(min_wavelength_sw, max_wavelength_sw, n_freq_sw)  # [m]
f_grid_sw = convert.wavelen2freq(wvl[::-1])  # [Hz]

min_wvn = 10  # [cm^-1]
max_wvn = 3210  # [cm^-1]
n_freq_lw = 200
wvn = np.linspace(min_wvn, max_wvn, n_freq_lw)
f_grid_lw = convert.kaycm2freq(wvn)
LW_flux_simulator = fsm.FluxSimulator("beach"+ "_LW")
species = [
            "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
            "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "CO2, CO2-CKDMT252",
            "O3",
            "O3-XFIT",
        ]
LW_flux_simulator.ws.f_grid = f_grid_lw
LW_flux_simulator.set_species(species)

SW_flux_simulator = fsm.FluxSimulator("beach" + "_SW")
SW_flux_simulator.ws.f_grid = f_grid_sw
SW_flux_simulator.emission = 0
SW_flux_simulator.gas_scattering = True


def get_atms_grd(ds):
    atms_grd = pyarts.arts.ArrayOfGriddedField4()
    for i in range(ds.sonde.size):
        profile = ds.isel(sonde=i)
        profile_grd = fsm.generate_gridded_field_from_profiles(
        profile["p"].values,
        profile["ta"].values,
        gases={
            "H2O": ph.specific_humidity2vmr(profile["q"]),
            "CO2": np.repeat(rh.gases["co2"], profile["q"].shape),
            "O3": profile["o3"],
            "N2": np.repeat(rh.gases["n2"], profile["q"].shape),
            "O2": np.repeat(rh.gases["o2"], profile["q"].shape),
            "CH4": np.repeat(rh.gases["ch4"], profile["q"].shape),
        },
        z_field=profile["altitude"].values,
    )
        atms_grd.append(profile_grd)
    LW_flux_simulator.get_lookuptableBatch(atms_grd)
    return atms_grd, LW_flux_simulator



def create_ds(ds):
    shape_lw_flux = (len(ds.sonde), len(ds.altitude), len(f_grid_lw))
    shape_sw_flux = (len(ds.sonde), len(ds.altitude), len(f_grid_sw) )
    shape_integrated = (len(ds.sonde), len(ds.altitude))

    fluxes = xr.Dataset(
        {
            "lw_flux_up_spectral": (
                ("sonde", "altitude", "f_grid_lw"),
                np.full(shape_lw_flux, np.nan),
            ),
            "lw_flux_down_spectral": (
                ("sonde", "altitude", "f_grid_lw"),
                np.full(shape_lw_flux, np.nan),
            ),
            "lw_flux_up": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "lw_flux_down": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "lw_heating_rate": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),

            "sw_flux_up_spectral": (
                            ("sonde", "altitude", "f_grid_sw"),
                            np.full(shape_sw_flux, np.nan),
                        ),
            "sw_flux_down_spectral": (
                ("sonde", "altitude", "f_grid_sw"),
                np.full(shape_sw_flux, np.nan),
            ),
            "sw_flux_up": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "sw_flux_down": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "sw_heating_rate": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
        },
        coords={
            "launch_lat": ("sonde", ds.launch_lat.values),
            "launch_lon": ("sonde", ds.launch_lon.values),
            "launch_time": ("sonde", ds.launch_time.values),
            "altitude": ("altitude", ds.altitude.values),
            "f_grid_lw": ("f_grid_lw", f_grid_lw),
            "f_grid_sw": ("f_grid_sw", f_grid_sw),
        },
    )
    return xr.merge([fluxes, ds], compat="override")

def init_store(store, ds):
    flxs = create_ds(ds)
    _, _ = get_atms_grd(ds)
    print(flxs)

    flxs.to_zarr(
        store, 
        encoding=rh.get_encoding(flxs),
        mode="w",
        compute=False,
        zarr_format=2
    )
    flxs[
            [
                "launch_time",
                "launch_lat",
                "launch_lon",
                "altitude",
             #   "latitude",
             #   "longitude",
                "f_grid_lw",
                "f_grid_sw",
                "o3",
                
            ]
        ].to_zarr(
            store,
            mode="r+",
        )



def calc_fluxes(ds):
    atms_grd, LW_flux_simulator = get_atms_grd(ds)
    flxs = create_ds(ds)
    start_time = time.time()

    for i in range(ds.sonde.size):
        sname = ds.sonde.isel(sonde=i).values
        profile = ds.isel(sonde=i)
        lat = profile.launch_lat.values
        lon = profile.launch_lon.values
        surface_temp = profile.ta.sel(altitude=0, method="nearest").values
        sun_pos = ph.get_arts_sun_pos(profile.launch_time.values)
        SW_flux_simulator.set_sun(sun_pos)
        lw = LW_flux_simulator.flux_simulator_single_profile(
            atms_grd[i],
            surface_temp,
            surface_altitude,
            surface_reflectivity_lw,
            geographical_position=[lat, lon],
        )
        sw = SW_flux_simulator.flux_simulator_single_profile(
            atms_grd[i],
            surface_temp,
                surface_altitude,
                surface_reflectivity_sw,
                geographical_position=[lat, lon],
            )
        for name, data in [("lw", lw), ("sw", sw)]:
            flxs[f"{name}_flux_up_spectral"].loc[dict(sonde=sname)] = data["spectral_flux_clearsky_up"].T
            flxs[f"{name}_flux_down_spectral"].loc[dict(sonde=sname)] = data["spectral_flux_clearsky_down"].T
            flxs[f"{name}_flux_up"].loc[dict(sonde=sname)] = data["flux_clearsky_up"]
            flxs[f"{name}_flux_down"].loc[dict(sonde=sname)] = data["flux_clearsky_down"]
            flxs[f"{name}_heating_rate"].loc[dict(sonde=sname)] = data["heating_rate_clearsky"]
        elapsed = time.time() - start_time
        remaining = elapsed / (i + 1) * (ds.sonde.size - i - 1)
        print(
                    f"{i + 1}/{ds.sonde.size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)",
                    flush=True,
                )
    return flxs

       

def write_region(store, region):
    dask.config.set(num_workers=32, scheduler="threads")

    ds = xr.open_dataset(sondes_for_rad).dropna(dim="sonde", how="any", subset=["ta", "q", "p", "o3", "altitude"]).isel(region)
    flxs = calc_fluxes(ds)

    flxs.drop_vars(
        [
            "launch_time",
            "launch_lat",
            "launch_lon",
            "altitude",
            "latitude",
            "longitude",
            "valid_time",
            "f_grid_lw",
            "f_grid_sw",
            "o3",
            "T",
            "P0",
            "theta", 
            "theta_rho", 
            "rh" 
        ], 
        errors="ignore"
    ).to_zarr(
        store,
        mode="r+",
        region=region,
    )


def _main():
    parser = argparse.ArgumentParser(description="Create arts fluxes")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--batchsize", default=10, type=int)
    parser.add_argument("-s", "--store", type=str)

    args = parser.parse_args()

    if args.init:
        ds = xr.open_dataset(sondes_for_rad).dropna(dim="sonde", how="any", subset=["ta", "q", "p", "o3", "altitude"])
        init_store(args.store, ds)
    else:
        batch_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        region = {
            "sonde": slice(batch_id * args.batchsize, (batch_id + 1) * args.batchsize)
        }
        print(batch_id, region, flush=True)
        print(args.store)
        write_region(args.store, region)


if __name__ == "__main__":
    _main()

# %%
