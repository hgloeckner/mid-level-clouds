import numpy as np
from xhistogram.xarray import histogram


def get_hist_of_ta(da_t, da_var, bins_var, bins_ta=np.linspace(240, 305, 200)):
    bin_name = da_var.name + "_bin"
    ta_name = da_t.name
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim=f"{ta_name}_bin")
        .rename({f"{ta_name}_bin": ta_name})
    )


def get_ml_cloud(wales, var="bsrgl", threshold=20, ml_min=4000, ml_max=8000):
    mid_level = wales.sel(altitude=slice(ml_min, ml_max))

    return (
        wales.where((mid_level[var].max(dim="altitude") >= threshold))
        .compute()
        .dropna("time", how="all")
    )
