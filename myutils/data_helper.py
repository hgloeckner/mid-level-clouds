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
    empty_value = xr.Dataset(
        data_vars={
            var: (("sonde_id", "altitude"), np.full((len(ds.sonde_id), 1), 0)),
            "ta": (("sonde_id", "altitude"), np.full((len(ds.sonde_id), 1), 0)),
        },
        coords={
            "sonde_id": ds.sonde_id,
            "altitude": [-1],
        },
    )

    interpolated_ds = (
        xr.concat(
            [
                empty_value,
                ds[[var, "ta"]].interpolate_na(dim="altitude"),
            ],
            dim="altitude",
        )
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
        lengths = start.values - end.values
        segments[str(sonde.values)] = (
            np.array(lengths),
            start.values,
            end.values,
        )
    return segments
