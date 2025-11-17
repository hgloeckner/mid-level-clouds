import xarray as xr
import numpy as np
import hashlib
import intake


def hash_xr_var(da):
    return np.array(
        [
            hashlib.sha256(str(entry).encode("ascii")).hexdigest()[:16]
            for entry in da.values
        ]
    )


def open_dropsondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return ds.reset_coords(["launch_altitude"])


def open_radiosondes(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.rename(
            {
                "height": "altitude",
                "platform": "platform_id",
            }
        )
        .reset_coords(["p", "lat", "lon", "interpolated_time", "sonde_id", "alt"])
        .swap_dims({"launch_time": "sonde"})
    )


def open_gate(cid):
    ds = xr.open_dataset(f"ipfs://{cid}", engine="zarr")
    return (
        ds.set_coords(["launch_lat", "launch_lon", "launch_time"])
        .swap_dims({"sonde": "launch_time"})
        .sel(launch_time=slice("1974-08-10", "1974-09-30"))
        .swap_dims({"launch_time": "sonde"})
    )


def open_reanalysis(chunks=None, **kwargs):
    if chunks is None:
        chunks = {}
    cat = intake.open_catalog("http://data.nextgems-h2020.eu/catalog.yaml")
    return {
        "ERA5": cat.ERA5(chunks=chunks, **kwargs).to_dask(),
        "MERRA2": cat.MERRA2(chunks=chunks, **kwargs).to_dask(),
        "JRA3Q": cat.JRA3Q(chunks=chunks, **kwargs).to_dask(),
    }


def get_cids():
    orcestra_main = "QmXkSUDo97PaDxsPzCPXJXwCFDLBMp7AVdPdV5CBQoagUN"
    return {
        "gate": "QmWFfuLW7VSqEFrAwaJr1zY9CzWqF4hC22yqgXELmY133K",
        "orcestra": orcestra_main,
        "radiosondes": f"{orcestra_main}/products/Radiosondes/Level_2/RAPSODI_RS_ORCESTRA_level2.zarr",
        "dropsondes": f"{orcestra_main}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    }


def open_wales(masked=True):
    if masked:
        wv_path = "/work/mh0066/m301046/Data/wales/wales_wv_masked.zarr"
        no_wv_path = "/work/mh0066/m301046/Data/wales/wales_no_wv_masked.zarr"
    else:
        wv_path = "/work/mh0066/m301046/Data/wales/wales_wv.zarr"
        no_wv_path = "/work/mh0066/m301046/Data/wales/wales_no_wv.zarr"

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


def get_gate_cid():
    return "QmeAFUdB3PZHRtCd441HjRGZPmEadtskXsnL34C9xigH3A"
