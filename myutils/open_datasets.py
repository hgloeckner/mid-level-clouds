import xarray as xr
import numpy as np
import hashlib


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "aircraft_latitude": "launch_lat",
                "aircraft_longitude": "launch_lon",
            }
        )
        .reset_coords(["aircraft_msl_altitude"])
        .swap_dims({"sonde": "sonde_id"})
    )


def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "alt": "altitude",
                "launch_time": "sonde_time",
                "sounding": "sonde_id",
                "flight_time": "bin_average_time",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "flight_lat", "flight_lon", "bin_average_time"])
        .swap_dims({"sonde_time": "sonde_id"})
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
                "time": "sonde_time",
            }
        )
        .set_coords(["launch_lat", "launch_lon"])
        .swap_dims({"sonde_time": "sonde_id"})
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


def get_cid():
    return "QmcAtU5Exu5z6xyPrQq3jm6jzG7aquHJPNMcew4MKAYNYQ"
