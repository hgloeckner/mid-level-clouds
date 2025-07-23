import numpy as np
from xhistogram.xarray import histogram
import xarray as xr


def get_hist_of_ta(da_t, da_var, bins_var, bins_ta=np.linspace(240, 305, 200)):
    bin_name = da_var.name + "_bin"
    ta_name = da_t.name
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim=f"{ta_name}_bin")
        .rename({f"{ta_name}_bin": ta_name})
    )


def get_hist_of_ta_2d(da_t, da_var, bins_var, bins_ta=np.linspace(240, 305, 200)):
    return histogram(
        da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude", "sonde_id"]
    )


def get_ml_cloud(wales, var="bsrgl", threshold=20, ml_min=4000, ml_max=8000):
    mid_level = wales.sel(altitude=slice(ml_min, ml_max))

    return (
        wales.where((mid_level[var].max(dim="altitude") >= threshold))
        .compute()
        .dropna("time", how="all")
    )


def sel_gate_region(
    gate, rs=None, ds=None, ascent_flag=0, lats=(5, 12), lons=(-27, -20)
):
    """
    ascent_flag: 0 for descending, 1 for ascending
    """
    if rs is not None:
        orcestra_gate = xr.concat(
            [
                rs.where(
                    (lons[0] < rs.launch_lon)
                    & (rs.launch_lon < lons[1])
                    & (lats[0] < rs.launch_lat)
                    & (rs.launch_lat < lats[1])
                    & (rs.ascent_flag == ascent_flag),
                    drop=True,
                ),
                ds.where(
                    (lons[0] < ds.launch_lon)
                    & (ds.launch_lon < lons[1])
                    & (lats[0] < ds.launch_lat)
                    & (ds.launch_lat < lats[1]),
                    drop=True,
                ),
            ],
            dim="sonde_id",
        )
    gate_region = gate.where(
        (lons[0] < gate.launch_lon)
        & (gate.launch_lon < lons[1])
        & (lats[0] < gate.launch_lat)
        & (gate.launch_lat < lats[1]),
        drop=True,
    )
    return (orcestra_gate, gate_region) if rs is not None else gate_region
