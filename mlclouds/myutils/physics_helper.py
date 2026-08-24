import astropy
import numpy as np
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.utilities as mtu
import moist_thermodynamics.functions as mtf
from moist_thermodynamics.saturation_vapor_pressures import es
import xarray as xr


def calc_Tv(T, mr):
    """
    Calculate the virtual temperature (Tv) from temperature (T) and mixing ratio (mr).
    Tv = T * (1 + 0.61 * mr)
    """
    eps = mtc.eps1  # kg / kg
    return T * (1 + mr / eps) / (1 + mr)


def get_csc_stab(rho, stability, H):
    cp = mtc.cpd
    Q = H * (cp * rho)
    grad_stability = stability.differentiate("altitude")
    return (
        -Q / cp / (stability**2) * grad_stability
    )  # 1 / (stability) ** 2 / rho * H * grad_stability


def get_csc_cooling(rho, stability, H):
    cp = mtc.cpd
    Q = H * (cp * rho)
    grad_H = Q.differentiate("altitude")
    return 1 / stability / cp * grad_H


def get_stability(theta, T):
    # \Gamma_d - \Gamma according to Holton & Hakim (5th edition)  2013 eq 2.49
    return T / theta * theta.differentiate("altitude")


def get_n2(th, qv, altdim="altitude"):
    """Returns the Brunt-Vaisala frequeny for unsaturated air.

    It assumes that the input are type xarray with their first coordinate being
    altitude in meters, and that the air is no where saturated

    Args:
        th: potential temperature
        qv: specific humidity
    """

    Rv = mtc.water_vapor_gas_constant
    Rd = mtc.dry_air_gas_constant
    g = mtc.gravity_earth
    R = Rd + (Rv - Rd) * qv
    dlnthdz = np.log(th).differentiate(altdim)
    dqvdz = qv.differentiate(altdim)

    return np.sqrt(g * (dlnthdz + (Rv - Rd) * dqvdz / R))


def specific_humidity2vmr(q):
    r"""Convert specific humidity to volume mixing ratio.

    .. math::
        x = \frac{q}{(1 - q) \frac{M_w}{M_d} + q}

    Parameters:
        q (float or ndarray): Specific humidity.

    Returns:
        float or ndarray: Volume mixing ratio.

    Examples:
        >>> specific_humidity2vmr(0.02)
        0.03176931009073226
    """
    Md = mtc.md
    Mw = mtc.molar_mass_h2o

    return q / ((1 - q) * Mw / Md + q)


def vmr2specific_humidity(x):
    r"""Convert volume mixing ratio to specific humidity.

    .. math::
        q = \frac{x}{(1 - x) \frac{M_d}{M_w} + x}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Specific humidity.

    Examples:
        >>> vmr2specific_humidity(0.04)
        0.025261087474946833
    """
    Md = mtc.md
    Mw = mtc.molar_mass_h2o

    return x / ((1 - x) * Md / Mw + x)


def mass_flux(rho, stability, H):
    return rho * H / (stability)


def density_from_q(p, T, q):
    Rd = mtc.dry_air_gas_constant
    Rv = mtc.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


def calc_heating_rate_from_flx(flx_up, flx_down, p):
    cp = mtc.cpd
    g = mtc.gravity_earth
    flx = flx_up - flx_down
    htg = g / cp * np.diff(flx, axis=-1) / np.diff(p, axis=-1)
    return np.insert(htg, -1, htg[-1])


def wv2q(wv):
    """
    get specific humidity from wales
    """
    m_l = mtc.atomic_mass_dry_air  # g/mol
    m_w = mtc.m_h2o  # g/mol water vapour molar mass
    C1 = m_w / m_l
    C2 = 1 - C1
    rho_w = wv.wv
    return C1 * rho_w / (wv.rho_air - C2 * rho_w)


def get_wdir_and_wspd(u, v):
    """
    Calculate wind direction and speed from u and v components.
    Wind direction is given in degrees from north.
    """
    wdir = (180 + np.arctan2(u, v) * 180 / np.pi) % 360
    wspd = np.sqrt(u**2 + v**2)
    return wdir, wspd



P = np.arange(100900.0, 4000.0, -500)
Rv = mtc.Rv
Rd = mtc.Rd


def make_sounding_from_adiabat(
    P, Tsfc=301.0, qsfc=17e-3, Tmin=200.0, thx=mtf.theta_l, integrate=False
) -> xr.Dataset:
    """creates a sounding from a moist adiabat

    Cacluates the moist adiabate based either on an integration or a specified
    isentrope with pressure as the vertical coordinate.

    Args:
        P: pressure
        Tsfc: starting (value at P.max()) temperature
        qsfc: starting (value at P.max()) specific humidity
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope if integrate = False
        integrate: determines if explicit integration will be used.
    """

    TPq = xr.Dataset(
        data_vars={
            "T": (
                ("levels",),
                mtu.moist_adiabat_with_ice(
                    P, Tx=Tsfc, qx=qsfc, Tmin=Tmin, thx=thx, integrate=integrate
                ),
                {"units": "K", "standard_name": "air_temperature", "symbol": "$T$"},
            ),
            "P": (
                ("levels",),
                P,
                {"units": "Pa", "standard_name": "air_pressure", "symbol": "$P$"},
            ),
            "q": (
                ("levels",),
                qsfc * np.ones(len(P)),
                {"units": "1", "standard_name": "specific_humidity", "symbol": "$q$"},
            ),
        },
    )
    TPq = TPq.assign(
        altitude=xr.DataArray(
            mtf.pressure_altitude(TPq.P, TPq.T, qv=TPq.q).values,
            dims=("levels"),
            attrs={
                "units": "m",
                "standard_name": "altitude",
                "description": "hydrostatic altitude given the datasets temperature and pressure",
            },
        )
    )
    TPq = TPq.assign(
        theta=(
            TPq.T.dims,
            mtf.theta(TPq.T, TPq.P).values,
            {
                "units": "K",
                "standard_name": "air_potential_teimerature",
                "symbol": "$\theta$",
            },
        )
    )
    TPq = TPq.assign(
        Trho=(
            TPq.T.dims,
            (
                TPq.T
                * (
                    1.0
                    - TPq.q
                    + mtf.saturation_partition(TPq.P, es(TPq.T), TPq.q) * Rv / Rd
                )
            ).values,
            {
                "units": "K",
                "standard_name": "density temperature",
                "symbol": "$T_\rho$",
            },
        )
    )
    TPq = TPq.assign(
        theta_rho=(
            TPq.T.dims,
            (
                TPq.theta
                * (
                    1.0
                    - TPq.q
                    + mtf.saturation_partition(TPq.P, es(TPq.T), TPq.q) * Rv / Rd
                )
            ).values,
            {
                "units": "K",
                "standard_name": "density potential temperature",
                "symbol": "$T_\rho$",
            },
        )
    )
    TPq = TPq.assign(
        P0=xr.DataArray(
            mtc.P0, attrs={"units": "Pa", "standards_name": "referenece_pressure"}
        )
    )

    return TPq.set_coords("altitude").swap_dims({"levels": "altitude"})

def get_arts_sun_pos(time):
    
    time = astropy.time.Time(time, scale="utc")

    sun = astropy.coordinates.get_sun(time)  # GCRS RA/Dec/distance
    distance_m = sun.distance.to(astropy.units.m).value

    gmst = time.sidereal_time("mean", "greenwich")  # in hourangle
    subsolar_lon = (sun.ra - gmst).wrap_at(180 * astropy.units.deg).deg
    subsolar_lat = sun.dec.deg

    return [distance_m, subsolar_lat, subsolar_lon]




def uniform_humidity(ds, zlcl, ztoa, rh, es=mtf.es_default):
    qrh = mtf.relative_humidity_to_specific_humidity(RH=rh, p=ds.p, T=ds.ta, es=es)
    qrh = qrh.where((qrh.altitude >= zlcl) & (qrh.altitude <= ztoa))
    return qrh.ffill(dim="altitude").bfill(dim="altitude")


def cshape_humidity(
    ds, zlcl, rhmid, rhlcl, rhtoa, Tmin=260, es=mtf.es_default, **kwargs
):
    rh = xr.DataArray(
        np.full_like(ds.ta.values, np.nan),
        dims=("altitude",),
        coords={"altitude": ds.altitude},
    )

    rh[ds.ta.argmin()] = rhtoa
    rh[np.abs(ds.ta - Tmin).argmin()] = rhmid
    rh[np.abs(ds.altitude - zlcl).argmin()] = rhlcl
    rh = rh.interpolate_na("altitude", method="quadratic")
    qrh = mtf.relative_humidity_to_specific_humidity(rh, ds.p, ds.ta, es=es)
    return qrh.ffill(dim="altitude").bfill(dim="altitude")


def eshape_humidity(
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
        np.full_like(ds.ta.values, np.nan),
        dims=("altitude",),
        coords={"altitude": ds.altitude},
    )

    rh[ds.ta.argmin()] = rhtoa
    rh[np.abs(ds.ta - Tmin).argmin()] = rhmid
    rh[np.abs(ds.altitude - zlcl).argmin()] = rhlcl
    rh = rh.interpolate_na("altitude", method="quadratic")
    rh = rh.where((ds.ta <= highlim) | (ds.ta >= lowlim))
    rh[np.abs(ds.ta - 273.15).argmin()] = (rhmid + rhlcl) * factor
    rh = rh.interpolate_na("altitude", method="quadratic")
    qrh = mtf.relative_humidity_to_specific_humidity(rh, ds.p, ds.ta, es=es)
    return qrh.ffill(dim="altitude").bfill(dim="altitude")
