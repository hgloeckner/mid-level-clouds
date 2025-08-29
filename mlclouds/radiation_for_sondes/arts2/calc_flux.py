#!/usr/bin/env python3
# SBATCH --account=mh0066
# SBATCH --partition=compute
# SBATCH --array=0-103%10
# SBATCH --time=00:30:00


import numcodecs
import numpy as np
import xarray as xr
import os
import argparse
import dask
import time

import pyarts
import FluxSimulator as fsm

idealized_sondes = "/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc"
min_wavenumber = 1
max_wavenumber = 3000
wave_bands = 10000


def get_chunks(sizes, chunksize=393216):
    match tuple(sizes.keys()):
        case ("sonde", "altitude", "f_grid"):
            sonde_chunksize = 10
            chunks = {
                "sonde": sonde_chunksize,
                "f_grid": chunksize // (10 * sonde_chunksize),
                "altitude": 10,
            }
        case ("sonde", "altitude"):
            sonde_chunksize = 10
            chunks = {
                "sonde": sonde_chunksize,
                "altitude": -1,
            }
        case (single_dim,):
            chunks = {
                single_dim: -1,
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    compressor = numcodecs.Blosc("zstd", shuffle=2, clevel=6)

    return {
        var: {
            "compressor": compressor,
            "dtype": "float32",
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
        if var != "sonde_id"
    }


def bitround(ds, keepbits=16, codec=None):
    def _bitround(var, keepbits, codec=None):
        if codec is None:
            codec = numcodecs.BitRound(keepbits=keepbits)

        return codec.decode(codec.encode(var))

    ds_rounded = xr.apply_ufunc(
        _bitround,
        ds,
        kwargs={"keepbits": keepbits},
        keep_attrs=True,
        dask="parallelized",
    )

    return ds_rounded


def init_calc(ds):
    atms_grd = pyarts.arts.ArrayOfGriddedField4()
    for i in range(ds.sonde.size):
        profile = ds.isel(sonde=i)

        profile_grd = fsm.generate_gridded_field_from_profiles(
            profile["p"].values,
            profile["t"].values,
            gases={
                "H2O": profile["H2O"],
                "CO2": profile["CO2"],
                "O3": profile["O3"],
                "N2": profile["N2"],
                "O2": profile["O2"],
            },
            z_field=profile["altitude"].values,
        )
        atms_grd.append(profile_grd)

    # Setup Flux Simulator
    f_grid = np.linspace(
        min_wavenumber, max_wavenumber, wave_bands
    )  # frequency grid in cm^-1
    f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)  # converted to Hz

    species = [
        "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        "CO2, CO2-CKDMT252",
        "O3",
        "O3-XFIT",
    ]

    flux_simulator = fsm.FluxSimulator(f"100m_1000m_{wave_bands}_wavebands")
    flux_simulator.ws.f_grid = f_grid_freq
    flux_simulator.set_species(species)
    flux_simulator.get_lookuptableBatch(atms_grd)
    return f_grid, flux_simulator, atms_grd


def create_ds(ds, f_grid):
    shape_flux = (len(ds.sonde), len(ds.altitude), len(f_grid))
    shape_integrated = (len(ds.sonde), len(ds.altitude))

    fluxes = xr.Dataset(
        {
            "lw_flux_up_spectral": (
                ("sonde", "altitude", "f_grid"),
                np.full(shape_flux, np.nan),
            ),
            "lw_flux_down_spectral": (
                ("sonde", "altitude", "f_grid"),
                np.full(shape_flux, np.nan),
            ),
            "lw_flux_up": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "lw_flux_down": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
            "heating_rate": (("sonde", "altitude"), np.full(shape_integrated, np.nan)),
        },
        coords={
            "sonde_id": ("sonde", ds.sonde_id.values),
            "altitude": ("altitude", ds.altitude.values),
            "f_grid": ("f_grid", f_grid),
        },
    )
    return xr.merge([fluxes.reset_coords(["sonde_id"]), ds], compat="override")


def initialize_store(store, ds):
    f_grid, _, _ = init_calc(ds)
    lw_fluxes = create_ds(ds, f_grid)
    lw_fluxes.to_zarr(
        store,
        encoding=get_encoding(lw_fluxes),
        mode="w",
        compute=False,
    )
    lw_fluxes[
        [
            "sonde_id",
            "altitude",
            "f_grid",
            "N2",
            "mag_w",
            "mag_u",
            "O3",
            "CO2",
            "wind_v",
            "wind_w",
            "wind_u",
            "O2",
            "mag_v",
        ]
    ].to_zarr(
        store,
        mode="r+",
    )


def calc_fluxes(ds):
    f_grid, flux_simulator, atms_grd = init_calc(ds)
    lw_fluxes = create_ds(ds, f_grid)
    # Run simulation and store results
    start_time = time.time()
    surface_reflectivity_lw = 0.05

    for i in range(ds.sonde.size):
        prof = ds.isel(sonde=i)
        surface_temp = prof.sel(altitude=0)["t"].item()

        result = flux_simulator.flux_simulator_single_profile(
            atms_grd[i],
            surface_temp,
            0.0,
            surface_reflectivity=surface_reflectivity_lw,
            z_field=prof["altitude"],
        )

        # Store spectral and integrated fluxes
        lw_fluxes["lw_flux_up_spectral"].loc[dict(sonde=i)] = result[
            "spectral_flux_clearsky_up"
        ].T
        lw_fluxes["lw_flux_down_spectral"].loc[dict(sonde=i)] = result[
            "spectral_flux_clearsky_down"
        ].T
        lw_fluxes["lw_flux_up"].loc[dict(sonde=i)] = result["flux_clearsky_up"]
        lw_fluxes["lw_flux_down"].loc[dict(sonde=i)] = result["flux_clearsky_down"]
        lw_fluxes["heating_rate"].loc[dict(sonde=i)] = result["heating_rate_clearsky"]

        # ETA logging
        elapsed = time.time() - start_time
        remaining = elapsed / (i + 1) * (ds.sonde.size - i - 1)
        print(
            f"{i + 1}/{ds.sonde.size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)",
            flush=True,
        )
    return lw_fluxes


def write_region(store, region):
    dask.config.set(num_workers=32, scheduler="threads")

    ds = xr.open_dataset(idealized_sondes).isel(region)
    lw_fluxes = calc_fluxes(ds)

    lw_fluxes.drop_vars(
        [
            "sonde_id",
            "altitude",
            "f_grid",
            "N2",
            "mag_w",
            "mag_u",
            "O3",
            "CO2",
            "wind_v",
            "wind_w",
            "wind_u",
            "O2",
            "mag_v",
        ]
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
        ds = xr.open_dataset(idealized_sondes)
        initialize_store(args.store, ds)
    else:
        batch_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        region = {
            "sonde": slice(batch_id * args.batchsize, (batch_id + 1) * args.batchsize)
        }
        print(batch_id, region, flush=True)
        write_region(args.store, region)


if __name__ == "__main__":
    _main()
