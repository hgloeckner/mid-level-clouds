#!/usr/bin/env python3
# SBATCH --account=mh0066
# SBATCH --partition=compute
# SBATCH --time=08:00:00


import FluxSimulator as fsm
import numpy as np
from pyarts.arts import convert


def build_frequency_grids() -> tuple[np.ndarray, np.ndarray]:
    """Builds the SW and LW frequency grids used by the flux simulators.

    Returns:
        A tuple of (f_grid_sw, f_grid_lw), each in Hz.
    """
    wvn_min_sw = 1 / 1e-5 / 100
    wvn_max_sw = 5e4
    n_wvn_sw = 100_000
    wvn_sw = np.logspace(np.log10(wvn_min_sw), np.log10(wvn_max_sw), n_wvn_sw)
    f_grid_sw = convert.kaycm2freq(wvn_sw)

    min_wvn = 10  # [cm^-1]
    max_wvn = 3250  # [cm^-1]
    n_freq_lw = 100_000
    wvn = np.linspace(min_wvn, max_wvn, n_freq_lw)
    f_grid_lw = convert.kaycm2freq(wvn)

    return f_grid_sw, f_grid_lw


def main() -> None:
    """Builds LW and SW lookup tables via FluxSimulator.get_lookuptableWide."""
    exp_name = "wn_100k"

    f_grid_sw, f_grid_lw = build_frequency_grids()

    species_lw = [
        "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        "CO2, CO2-CKDMT252",
        "O3",
        "O3-XFIT",
        "CH4",
    ]
    species_sw = [
        "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        "CO2, CO2-CKDMT252",
        "O3",
        "O3-XFIT",
    ]

    LW_flux_simulator = fsm.FluxSimulator(exp_name + "_LW")
    LW_flux_simulator.ws.f_grid = f_grid_lw
    LW_flux_simulator.set_species(species_lw)
    LW_flux_simulator.set_paths(
        lut_path="/work/mh0066/m301046/data/mlclouds/lookup_tables/LW"
    )

    LW_flux_simulator.LUT_wide_h2o_vmr_default_parameters = [
        1000.0,
        1e-08,
        100000.0,
        0.05,
    ]
    LW_flux_simulator.get_lookuptableWide(
        t_min=170.0,
        recalc=True,
    )

    SW_flux_simulator = fsm.FluxSimulator(exp_name + "_SW")
    SW_flux_simulator.ws.f_grid = f_grid_sw
    SW_flux_simulator.set_species(species_sw)
    SW_flux_simulator.set_paths(
        lut_path="/work/mh0066/m301046/data/mlclouds/lookup_tables/SW"
    )
    SW_flux_simulator.LUT_wide_h2o_vmr_default_parameters = [
        1000.0,
        1e-08,
        100000.0,
        0.05,
    ]
    SW_flux_simulator.get_lookuptableWide(
        t_min=170.0,
        recalc=True,
    )


if __name__ == "__main__":
    main()
