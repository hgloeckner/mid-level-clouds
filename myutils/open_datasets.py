import xarray as xr
import numpy as np
import hashlib
import intake


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    ds = (
        ds.rename(
            {
                "aircraft_latitude": "launch_lat",
                "aircraft_longitude": "launch_lon",
            }
        )
        .reset_coords(["aircraft_msl_altitude"])
        .swap_dims({"sonde": "sonde_id"})
    )
    try:
        return ds.swap_dims({"circle": "circle_id"})
    except ValueError:
        return ds


def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "flight_lat", "flight_lon", "bin_average_time"])
        .swap_dims({"launch_time": "sonde_id"})
    )


def hash_xr_var(da):
    return np.array(
        [
            hashlib.sha256(str(entry).encode("ascii")).hexdigest()[-8:]
            for entry in da.values
        ]
    )


def open_gate(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    ds = ds.assign_coords({"sonde_id": ("time", hash_xr_var(ds.time))})
    return (
        ds.rename(
            {
                "alt": "altitude",
                "lat_beg": "launch_lat",
                "lon_beg": "launch_lon",
                "ua": "u",
                "va": "v",
                "platforms": "platform_id",
                "time": "launch_time",
            }
        )
        .set_coords(["launch_lat", "launch_lon"])
        .swap_dims({"launch_time": "sonde_id"})
    )


def open_wales(masked=True):
    if masked:
        wv_path = "/work/mh0066/m301046/wales/wales_wv_masked.zarr"
        no_wv_path = "/work/mh0066/m301046/wales/wales_no_wv_masked.zarr"
    else:
        wv_path = "/work/mh0066/m301046/wales/wales_wv.zarr"
        no_wv_path = "/work/mh0066/m301046/wales/wales_no_wv.zarr"

    wv = xr.open_dataset(
        wv_path,
        engine="zarr",
        chunks={},
    )
    no_wv = xr.open_dataset(
        no_wv_path,
        engine="zarr",
        chunks={},
    )
    return wv.rename(
        {
            "airtemperature": "ta",
            "flight_altitude": "aircraft_msl_altitude",
            "airdensity": "rho_air",
        }
    ), no_wv.rename(
        {
            "airtemperature": "ta",
            "flight_altitude": "aircraft_msl_altitude",
            "airdensity": "rho_air",
        }
    )


def open_radiative_fluxes(path=None):
    if path is None:
        path = "/work/mh0066/m301046/ml_clouds/arts2_fluxes.zarr"
    ds = (
        xr.open_dataset(
            path,
            engine="zarr",
            chunks={},
        )
        .swap_dims({"sonde": "sonde_id"})
        .rename({"t": "ta"})
    )
    return ds.assign(cooling_rate=-ds.heating_rate)


def open_reanalysis(chunks=None, **kwargs):
    if chunks is None:
        chunks = {}
    cat = intake.open_catalog("http://data.nextgems-h2020.eu/catalog.yaml")
    return {
        "ERA5": cat.ERA5(chunks=chunks, **kwargs).to_dask(),
        "MERRA2": cat.MERRA2(chunks=chunks, **kwargs).to_dask(),
        "JRA3Q": cat.JRA3Q(chunks=chunks, **kwargs).to_dask(),
    }


def get_cid():
    return "QmPNVTb5fcN59XUi2dtUZknPx5HNnknBC2x4n7dtxuLdwi"
