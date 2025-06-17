import xarray as xr
import numpy as np
from xhistogram.xarray import histogram


def get_hist(ds, xvar, yvar, var, xbins, ybins):
    base_hist = histogram(
        ds[yvar].where(~np.isnan(ds[var])),
        ds[xvar].where(~np.isnan(ds[var])),
        bins=[ybins, xbins],
    )
    weights = (ds[var].where(~np.isnan(ds[var]))).fillna(0.0)
    return (
        histogram(
            ds[yvar].where(~np.isnan(ds[var])),
            ds[xvar].where(~np.isnan(ds[var])),
            bins=[ybins, xbins],
            weights=weights,
        )
        / base_hist
    )


def get_segments(
    ds,
    var,
    varmin,
):
    interpolated_ds = (
        ds[[var, "ta"]]
        .where((ds.altitude > ds.altitude.min()) & (ds.altitude < ds.altitude.max()))
        .interpolate_na(dim="altitude")
        .fillna(0)
        .load()
    )
    mask = xr.where(
        interpolated_ds[var] > varmin,
        1,
        0,
    )
    diff = mask.diff(dim="altitude")
    starts = diff.where(
        diff == 1,
    )

    ends = diff.where(
        diff == -1,
    )
    segments = {}
    for sonde in mask.sonde_id:
        start = ds.ta.sel(
            sonde_id=sonde,
            altitude=starts.sel(sonde_id=sonde).dropna(dim="altitude").altitude,
        )
        end = ds.ta.sel(
            sonde_id=sonde,
            altitude=ends.sel(sonde_id=sonde).dropna(dim="altitude").altitude,
        )

        lengths = end.altitude.values - start.altitude.values
        segments[str(sonde.values)] = (
            np.array(lengths),
            start.values,
            end.values,
        )
    return segments
