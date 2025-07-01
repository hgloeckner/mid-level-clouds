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


def get_cid():
    return "QmP73Kosem4exJcZXxG8vpN4YLqaepoZSPWwnQ9N1xffus"
