import numpy as np
import xarray as xr
import moist_thermodynamics.functions as mtf
from matplotlib.path import Path

east = [[-34, 3.5], [-20, 3.5], [-20, 13.5], [-34, 13.5]]
west = [[-59, 6], [-45, 6], [-45, 16], [-59, 16]]
gate_a = [
    [-27.0, 6.5],
    [-23.5, 5.0],
    [-20.0, 6.5],
    [-20.0, 10.5],
    [-23.5, 12.0],
    [-27.0, 10.5],
]
north = [[-26, 13.5], [-20, 13.5], [-20, 18.5], [-26, 18.5]]

variable_attribute_dict = {
    "ta": {
        "standard_name": "air_temperature",
        "units": "K",
    },
    "p": {
        "standard_name": "air_pressure",
        "units": "Pa",
    },
    "q": {
        "standard_name": "specific_humidity",
        "units": "kg/kg",
    },
    "u": {
        "standard_name": "eastward_wind",
        "units": "m/s",
    },
    "v": {
        "standard_name": "northward_wind",
        "units": "m/s",
    },
    "rh": {
        "standard_name": "relative_humidity",
        "units": "1",
        "description": "Relative to Wagner-Pruss saturation vapor pressure over liquid",
    },
    "theta": {
        "standard_name": "air_potential_temperature",
        "units": "K",
        "description": "Use dry air gas constants and 1000 hPa as reference pressure",
    },
}


def rolling_hist(ds, ta_bin_low=15, rh_bin_low=5, ta_bin_up=20, rh_bin_up=7):
    return xr.concat(
        [
            (
                ds.where(ds > 0)
                .rolling(ta_bin=ta_bin_low, center=True, min_periods=3)
                .sum()
                .rolling(rh_bin=rh_bin_low, center=True, min_periods=3)
                .sum()
                .sel(ta_bin=slice(270, 305))
            ),
            (
                ds.where(ds > 0)
                .rolling(ta_bin=ta_bin_up, center=True, min_periods=3)
                .sum()
                .rolling(rh_bin=rh_bin_up, center=True, min_periods=3)
                .sum()
                .sel(ta_bin=slice(None, 270))
            ),
        ],
        dim="ta_bin",
    ).sortby("ta_bin")


def interpolate_gaps(ds):
    akima_vars = ["u", "v"]
    linear_vars = ["theta", "q", "p"]

    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="akima", max_gap=1500)
            for var in akima_vars
        }
    )
    ds = ds.assign(p=np.log(ds.p))
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(dim="altitude", method="linear", max_gap=1500)
            for var in linear_vars
        }
    )
    ds = ds.assign(p=np.exp(ds.p))
    ds = ds.assign(
        ta=mtf.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mtf.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )

    return ds


def extrapolate_sfc(ds):
    """
    Extrapolate surface values to the lowest level.
    This function assumes that the dataset has an altitude dimension.
    """
    constant_vars = ["u", "v", "theta", "q"]
    ds = ds.assign(
        **{
            var: ds[var].interpolate_na(
                dim="altitude", method="nearest", max_gap=300, fill_value="extrapolate"
            )
            for var in constant_vars
        }
    )
    ds = ds.assign(
        p=np.exp(
            np.log(ds.p).interpolate_na(
                dim="altitude", method="linear", max_gap=300, fill_value="extrapolate"
            )
        )
    )
    ds = ds.assign(
        ta=mtf.theta2T(ds.theta, ds.p),
    )
    ds = ds.assign(
        rh=mtf.specific_humidity_to_relative_humidity(ds.q, ds.p, ds.ta),
    )
    return ds


def sel_sub_domain(
    ds, polygon, item_var="sonde", lon_var="launch_lon", lat_var="launch_lat"
):
    """
    select points from dataset that lie within the polygon
    """
    points = np.column_stack([ds[lon_var].values, ds[lat_var].values])
    inside = Path(polygon).contains_points(points)
    return ds.sel(**{item_var: inside})


def find_highest_cloud_altitude(
    ds, variable_name="bsrgl", threshold=20, altdim="altitude"
):
    ds = ds.sortby(altdim)  # .chunk({"altitude": -1, "time": 1000})
    mask = ds[variable_name] >= threshold
    mask_inv = mask.isel({altdim: slice(None, None, -1)})

    highest_altitude = ds[altdim].values[
        mask.sizes[altdim] - 1 - mask_inv.argmax(dim=altdim)
    ]
    return np.where(mask.any(dim=altdim), highest_altitude, np.nan)
