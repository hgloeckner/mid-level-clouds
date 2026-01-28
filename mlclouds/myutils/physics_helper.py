import numpy as np
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.utilities as mtu
import moist_thermodynamics.functions as mtf
import xarray as xr


def calc_Tv(T, mr):
    """
    Calculate the virtual temperature (Tv) from temperature (T) and mixing ratio (mr).
    Tv = T * (1 + 0.61 * mr)
    """
    eps = mtc.eps1  # kg / kg
    return T * (1 + mr / eps) / (1 + mr)


def get_stability(theta, T):
    # \Gamma_d - \Gamma according to Holton & Hakim (5th edition)  2013 eq 2.49
    return (T / theta * theta.differentiate("altitude")) * 1000


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


def get_csc_stab(rho, stability, H):
    grad_stability = stability.differentiate("altitude") * 1000
    return 1 / (stability) * (H / stability * grad_stability)


def get_csc_cooling(rho, stability, H):
    grad_H = H.differentiate("altitude") * 1000
    return -1 / (stability) * grad_H


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
        P0=xr.DataArray(
            mtc.P0, attrs={"units": "Pa", "standards_name": "referenece_pressure"}
        )
    )

    return TPq.set_coords("altitude").swap_dims({"levels": "altitude"})
