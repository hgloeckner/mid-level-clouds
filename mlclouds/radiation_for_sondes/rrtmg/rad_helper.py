import numpy as np
import xarray as xr
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.functions as mtf


def make_atmosphere(p, T, h2o_vmr, o3, T_s=None):
    """Create a pyRTE-RRTMG atmosphere from pressure, temperature and humidity arrays."""
    if np.any(p[..., 0] < p[..., -1]):
        raise ValueError("Arrays need to be passed in ascending order")

    if T_s is None:
        T_s = T[..., 0]

    atmosphere = xr.Dataset(
        data_vars={
            "pres_level": (("column", "level"), p),
            "temp_level": (("column", "level"), T),
            "pres_layer": (("column", "layer"), 0.5 * (p[..., 1:] + p[..., :-1])),
            "temp_layer": (("column", "layer"), 0.5 * (T[..., 1:] + T[..., :-1])),
            "surface_temperature": (("column",), T_s),
            "h2o": (("column", "layer"), 0.5 * (h2o_vmr[..., 1:] + h2o_vmr[..., :-1])),
            "o3": (("layer"), 0.5 * (o3[..., 1:] + o3[..., :-1])),
            "co2": 422e-6,
            "ch4": 1650e-9,
            "n2o": 306e-9,
            "n2": 0.7808,
            "o2": 0.2095,
            "co": 0.0,
        },
    )

    return atmosphere


def calc_heating_rate_from_flx(flx_up, flx_down, p):
    cp = mtc.cpd
    g = mtc.gravity_earth
    flx = flx_up - flx_down
    htg = g / cp * np.diff(flx, axis=-1) / np.diff(p, axis=-1)
    return np.insert(htg, -1, htg[-1])


def uniform_humidity(ds, zlcl, ztoa, rh, es=mtf.es_default):
    qrh = mtf.relative_humidity_to_specific_humidity(RH=rh, p=ds.P, T=ds.T, es=es)
    qrh = qrh.where((qrh.altitude >= zlcl) & (qrh.altitude <= ztoa))
    return qrh.ffill(dim="altitude").bfill(dim="altitude")


def cshape_humidity(
    ds, zlcl, rhmid, rhlcl, rhtoa, Tmin=260, es=mtf.es_default, **kwargs
):
    rh = xr.DataArray(
        np.full_like(ds.T.values, np.nan),
        dims=("altitude",),
        coords={"altitude": ds.altitude},
    )

    rh[ds.T.argmin()] = rhtoa
    rh[np.abs(ds.T - Tmin).argmin()] = rhmid
    rh[np.abs(ds.altitude - zlcl).argmin()] = rhlcl
    rh = rh.interpolate_na("altitude", method="quadratic")
    qrh = mtf.relative_humidity_to_specific_humidity(rh, ds.P, ds.T, es=es)
    return qrh.ffill(dim="altitude").bfill(dim="altitude")


def wshape_humidity(
    ds,
    zlcl,
    rhmid,
    rhlcl,
    rhtoa,
    lowlim=280,
    highlim=265,
    factor=0.5,
    Tmin=260,
    es=mtf.es_default,
):
    rh = xr.DataArray(
        np.full_like(ds.T.values, np.nan),
        dims=("altitude",),
        coords={"altitude": ds.altitude},
    )

    rh[ds.T.argmin()] = rhtoa
    rh[np.abs(ds.T - Tmin).argmin()] = rhmid
    rh[np.abs(ds.altitude - zlcl).argmin()] = rhlcl
    rh = rh.interpolate_na("altitude", method="quadratic")
    rh = rh.where((ds.T <= highlim) | (ds.T >= lowlim))
    rh[np.abs(ds.T - 273.15).argmin()] = (rhmid + rhlcl) * factor
    rh = rh.interpolate_na("altitude", method="quadratic")
    qrh = mtf.relative_humidity_to_specific_humidity(rh, ds.P, ds.T, es=es)
    return qrh.ffill(dim="altitude").bfill(dim="altitude")
