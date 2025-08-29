import numpy as np
import moist_thermodynamics.constants as constants


def get_freezing_levels(ds, vars=None):
    if vars is None:
        vars = ["rh", "u", "v", "tv", "mse", "vmse", "umse"]

    swap_d = ds.swap_dims({"sonde": "sonde_id"})
    indices = (
        (np.abs(swap_d.ta.interpolate_na(dim="altitude") - 273.15))
        .dropna(dim="sonde_id", how="all")
        .argmin(dim="altitude")
    )

    for variable in vars:
        d = []
        varlist = []
        for sonde_id in swap_d.sonde_id:
            try:
                alt = ds.altitude[indices.sel(sonde_id=sonde_id)].values
                d.append(alt)
            except KeyError:
                d.append(np.nan)
                varlist.append(np.nan)
            else:
                varlist.append(swap_d.sel(altitude=alt, sonde_id=sonde_id)[variable])
        swap_d = swap_d.assign({f"freezing_{variable}": (("sonde_id"), varlist)})
    swap_d = swap_d.assign(freezing_level=(("sonde_id"), d))
    return swap_d.swap_dims({"sonde_id": "sonde"})


def calc_Tv(T, mr):
    """
    Calculate the virtual temperature (Tv) from temperature (T) and mixing ratio (mr).
    Tv = T * (1 + 0.61 * mr)
    """
    eps = constants.eps1  # kg / kg
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

    Rv = constants.water_vapor_gas_constant
    Rd = constants.dry_air_gas_constant
    g = constants.gravity_earth
    R = Rd + (Rv - Rd) * qv
    dlnthdz = np.log(th).differentiate(altdim)
    dqvdz = qv.differentiate(altdim)

    return np.sqrt(g * (dlnthdz + (Rv - Rd) * dqvdz / R))


def get_csc_stab(rho, stability, H):
    grad_stability = stability.differentiate("altitude") * 1000
    cp = constants.cpv
    return 1 / (cp * rho * stability) * (H / stability * grad_stability)


def get_csc_cooling(rho, stability, H):
    grad_H = H.differentiate("altitude") * 1000
    cp = constants.cpv
    return -1 / (cp * rho * stability) * grad_H


def density_from_q(p, T, q):
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


def wv2q(wv):
    """
    get specific humidity from wales
    """
    m_l = constants.atomic_mass_dry_air  # g/mol
    m_w = constants.m_h2o  # g/mol water vapour molar mass
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
