"""
Cumulative gasification cases: 1 … 10 largest plants in plants_clean (by m_tot).

Each case replaces 1…10 decentralized waste-CHP sites with one centralized gasifier.
Per replaced site: truck haul to the hub + heat-pump CAPEX/OPEX (gasification.py).
Central hub: sorting, gasification, H₂ — one scaled CAPEX/OPEX block on aggregated waste.
All cost components are levelized at case end on total methanol [t/a]. Three summary tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from model import compression_energy, get_CoolProp

# --- Physical / economic constants (aligned with gasification.py) ---
ETA_BOILER = 0.85  # [-] boiler efficiency
RW_EVAP_MJ_PER_KG = 2.5  # [MJ/kg] latent heat for water evaporation (fuel moisture)
LHV_H2_MJ_PER_KMOL = 243.0  # [MJ/kmolH2] lower heating value of H2
LHV_CO_MJ_PER_KMOL = 286.0  # [MJ/kmolCO] lower heating value of CO
FLH_H_PER_YR = 8000.0  # [h/yr] full-load hours for nameplate rates
ETA_ELECTROLYZER = 0.699  # [-] MW H2 (LHV) / MW el — stack + BOP efficiency anchor

FIXATE_CAPEX = 0.05  # [-] fixed OPEX as fraction of overnight CAPEX per block
SCALE_EXPONENT_K = 0.67  # [-] installed cost scaling exponent (capacity ratio^k)
SEK_TO_EUR = 0.091  # [EUR/SEK] currency
DISCOUNT_RATE = 0.075  # [-] real discount rate for CRF
LIFETIME_YR = 25  # [yr] economic lifetime for levelization
RECYCLE_RATIO = 3  # [-] compressor work multiplier for recycle loops

CAPEX_SORTING_REF_MEUR = 350.0 * SEK_TO_EUR  # [MEUR] reference sorting CAPEX at reference waste throughput
CAPACITY_SORTING_REF_T_PER_YR = 140_000.0  # [t waste/a] reference sorting capacity (Brista)

CAPEX_GASIFICATION_REF_MEUR = 749_729_639 * 1e-6  # [MEUR] reference gasification+synfuel train (ECOPLANTA)
CAPACITY_GASIFICATION_REF_T_PER_YR = 237_000.0  # [t methanol/a] reference methanol nameplate

CAPEX_H2_REF_KEUR_PER_MWE = 550.0  # [kEUR/MWel] AEC reference specific cost (100 MW class)
CELC_EUR_PER_MWH = 50.0  # [EUR/MWh_el] variable electricity price for OPEX

OPEX_VAR_SORTING_EUR_PER_T_WASTE = 200.0 * SEK_TO_EUR  # [EUR/t waste] variable sorting OPEX (Brista)
OPEX_VAR_GASIFICATION_EUR_PER_MWH_FUEL = 1.4  # [EUR/MWh_fuel] variable non-energy OPEX (Beiron 2026)

COP_HEAT_PUMP = 2.5  # [-] heat pump COP (gasification.py)
CAPEX_HP_REF_MEUR_PER_MWTH = 0.86  # [MEUR/MWth] [Bergander & Hellander, 2024]

N_STEAM_ASSUMED = 1.0  # [kmolH2O/kmolC] steam moles assumed in gasification step [R3]
AIR_RATIO_COMBUSTOR = 1.2  # [-] excess air for O2 demand to combustor

N_CASES = 10  # [-] number of cumulative plant-count scenarios (largest … 10th largest)


@dataclass
class FuelAggregate:
    """Summed extensive properties for plants in the cumulative set (single virtual fuel)."""

    n_c_pl: float  # [kmolC_pl/yr] fossil carbon in aggregate feed
    n_h_pl: float  # [kmolH_pl/yr]
    n_o_pl: float  # [kmolO_pl/yr]
    n_c_bio: float  # [kmolC_bio/yr]
    n_h_bio: float  # [kmolH_bio/yr]
    n_o_bio: float  # [kmolO_bio/yr]
    m_pl: float  # [kg_pl/yr]
    m_bio: float  # [kg_bio/yr]
    m_h2o: float  # [kg_h2o/yr]
    m_ash: float  # [kg_ash/yr]
    m_tot: float  # [kg_waste/yr] total mass throughput (wet)
    e_input_mj_yr: float  # [MJ/yr] Σ(LHV_pl·m_pl + LHV_bio·m_bio) dry-fuel chemical energy
    lhv_tot_blend: float  # [MJ/kg_wet] mass-weighted mean LHV_tot for OPEX fuel-energy term


def aggregate_fuel(plants_slice: pd.DataFrame) -> FuelAggregate:
    """Sum extensive columns over ``plants_slice``; blended LHV_tot from Σ(LHV_tot·m_tot)/Σm_tot."""
    n_c_pl = float(plants_slice["nC_pl"].sum())  # [kmolC_pl/yr]
    n_h_pl = float(plants_slice["nH_pl"].sum())  # [kmol/yr]
    n_o_pl = float(plants_slice["nO_pl"].sum())  # [kmol/yr]
    n_c_bio = float(plants_slice["nC_bio"].sum())  # [kmol/yr]
    n_h_bio = float(plants_slice["nH_bio"].sum())  # [kmol/yr]
    n_o_bio = float(plants_slice["nO_bio"].sum())  # [kmol/yr]
    m_pl = float(plants_slice["m_pl"].sum())  # [kg/yr]
    m_bio = float(plants_slice["m_bio"].sum())  # [kg/yr]
    m_h2o = float(plants_slice["m_h2o"].sum())  # [kg/yr]
    m_ash = float(plants_slice["m_ash"].sum())  # [kg/yr]
    m_tot = float(plants_slice["m_tot"].sum())  # [kg/yr]
    e_input = float(
        (plants_slice["LHV_pl"] * plants_slice["m_pl"] + plants_slice["LHV_bio"] * plants_slice["m_bio"]).sum()
    )  # [MJ/yr]
    lhv_num = float((plants_slice["LHV_tot"] * plants_slice["m_tot"]).sum())  # [MJ/yr] = Σ(LHV_tot·m_tot)
    lhv_tot_blend = lhv_num / m_tot if m_tot > 0 else 0.0  # [MJ/kg_wet]
    return FuelAggregate(
        n_c_pl=n_c_pl,
        n_h_pl=n_h_pl,
        n_o_pl=n_o_pl,
        n_c_bio=n_c_bio,
        n_h_bio=n_h_bio,
        n_o_bio=n_o_bio,
        m_pl=m_pl,
        m_bio=m_bio,
        m_h2o=m_h2o,
        m_ash=m_ash,
        m_tot=m_tot,
        e_input_mj_yr=e_input,
        lhv_tot_blend=lhv_tot_blend,
    )


def interpolate_truck_cost_eur_per_t(
    mass_t_per_yr,
    distance_km,
    truck_costs_df,
    sek_to_eur,
    debug=False,
):
    """2D interpolation: annual mass [t/a] and haul [km] → truck unit cost [EUR/t waste]."""
    masses = truck_costs_df["mass"].values  # [t/a]
    distances = truck_costs_df["km"].values  # [km]
    costs_sek_per_t = truck_costs_df["SEK/ton"].values  # [SEK/t]
    points = np.column_stack((masses, distances))
    mass_min, mass_max = masses.min(), masses.max()
    dist_min, dist_max = distances.min(), distances.max()
    method = (
        "nearest"
        if (
            mass_t_per_yr < mass_min
            or mass_t_per_yr > mass_max
            or distance_km < dist_min
            or distance_km > dist_max
        )
        else "linear"
    )
    cost_sek_per_t = float(griddata(points, costs_sek_per_t, (mass_t_per_yr, distance_km), method=method))
    cost_eur_per_t = cost_sek_per_t * sek_to_eur  # [EUR/t]
    if debug:
        print(
            "interpolate_truck_cost_eur_per_t:",
            f"mass={mass_t_per_yr:.0f} t/a",
            f"km={distance_km:.0f}",
            f"method={method}",
            f"cost={cost_eur_per_t:.2f} EUR/t",
        )
    return cost_eur_per_t


def replacement_costs_for_plant(plant_row: pd.Series, truck_costs_df: pd.DataFrame, debug: bool = False) -> Dict[str, float]:
    """Truck haul + heat-pump replacement for one decentralized plant shut down in favour of the hub.

    Returns overnight CAPEX [MEUR] and annual OPEX [MEUR/a] components (truck has no CAPEX).
    """
    name = plant_row["Name"]
    waste_mass_t_per_yr = float(plant_row["m_tot"]) / 1000.0  # [t/a]
    distance_km = float(plant_row["gasification_distance_km"])  # [km]
    flh_plant = float(plant_row["FLH"])  # [h/yr] site-specific full-load hours

    truck_eur_per_t = interpolate_truck_cost_eur_per_t(
        waste_mass_t_per_yr, distance_km, truck_costs_df, SEK_TO_EUR, debug=debug
    )
    opex_truck_eur_yr = waste_mass_t_per_yr * truck_eur_per_t  # [EUR/yr] variable transport only

    q_lost_mwth = float(plant_row["Qdh"]) + float(plant_row["Qfgc"])  # [MWth] heat output replaced
    p_lost_mwel = float(plant_row["P"])  # [MWel] retained in OPEX term as in gasification.py
    whp_mwel = q_lost_mwth / COP_HEAT_PUMP  # [MWel]
    capex_hp_meur = CAPEX_HP_REF_MEUR_PER_MWTH * q_lost_mwth  # [MEUR]
    opex_fix_hp_meur = capex_hp_meur * FIXATE_CAPEX  # [MEUR/a]
    opex_var_hp_meur = (whp_mwel + p_lost_mwel) * flh_plant * CELC_EUR_PER_MWH * 1e-6  # [MEUR/a]
    opex_hp_meur = opex_fix_hp_meur + opex_var_hp_meur  # [MEUR/a]

    fuel_mwth = float(plant_row["Qlhv"])  # [MWth] waste fuel LHV rate (characterize_plants)
    heat_out_mwth = q_lost_mwth  # [MWth] district heat + flue-gas condenser duty
    power_out_mwel = p_lost_mwel  # [MWel] CHP electricity export lost with shutdown

    if debug:
        print(f"replacement_costs_for_plant {name}:", truck_eur_per_t, "EUR/t", capex_hp_meur, "MEUR CAPEX HP")

    return {
        "plant_name": name,
        "opex_truck_eur_yr": opex_truck_eur_yr,  # [EUR/yr]
        "capex_hp_meur": capex_hp_meur,  # [MEUR]
        "opex_hp_meur": opex_hp_meur,  # [MEUR/a]
        "opex_fix_hp_meur": opex_fix_hp_meur,  # [MEUR/a]
        "opex_var_hp_meur": opex_var_hp_meur,  # [MEUR/a]
        "fuel_mwth": fuel_mwth,  # [MWth] before replacement
        "heat_out_mwth": heat_out_mwth,  # [MWth] before replacement
        "power_out_mwel": power_out_mwel,  # [MWel] before replacement
        "heat_deficit_mwth": heat_out_mwth,  # [MWth] load met by HP after shutdown
        "hp_elec_mwel": whp_mwel,  # [MWel] heat-pump electricity
        "power_deficit_mwel": whp_mwel + power_out_mwel,  # [MWel] total site electricity after replacement
    }


def sum_replacement_costs(plants_slice: pd.DataFrame, truck_costs_df: pd.DataFrame, debug: bool = False) -> Dict[str, float]:
    """Sum per-plant replacement costs over all plants in ``plants_slice`` (each plant replaced once)."""
    opex_truck_eur_yr = 0.0
    capex_hp_meur = 0.0
    opex_hp_meur = 0.0
    opex_fix_hp_meur = 0.0
    opex_var_hp_meur = 0.0
    per_plant: List[Dict[str, float]] = []
    for _, row in plants_slice.iterrows():
        part = replacement_costs_for_plant(row, truck_costs_df, debug=debug)
        per_plant.append(part)
        opex_truck_eur_yr += part["opex_truck_eur_yr"]
        capex_hp_meur += part["capex_hp_meur"]
        opex_hp_meur += part["opex_hp_meur"]
        opex_fix_hp_meur += part["opex_fix_hp_meur"]
        opex_var_hp_meur += part["opex_var_hp_meur"]
    return {
        "opex_truck_eur_yr": opex_truck_eur_yr,  # [EUR/yr]
        "capex_hp_meur": capex_hp_meur,  # [MEUR]
        "opex_hp_meur": opex_hp_meur,  # [MEUR/a]
        "opex_fix_hp_meur": opex_fix_hp_meur,  # [MEUR/a]
        "opex_var_hp_meur": opex_var_hp_meur,  # [MEUR/a]
        "per_plant": per_plant,
    }


def levelize_opex_eur_yr_to_eur_per_t_methanol(opex_eur_yr: float, annual_methanol_t_per_yr: float) -> float:
    """Annual OPEX in EUR/yr → [EUR/t methanol]."""
    return opex_eur_yr / annual_methanol_t_per_yr


def levelize_opex_meur_a_to_eur_per_t_methanol(opex_meur_a: float, annual_methanol_t_per_yr: float) -> float:
    """Annual OPEX in MEUR/a → [EUR/t methanol]."""
    return opex_meur_a * 1e6 / annual_methanol_t_per_yr


def levelize_all_costs(
    annual_methanol_t_per_yr: float,
    central: Dict[str, float],
    replacement: Dict[str, float],
    dr: float = DISCOUNT_RATE,
    lifetime: int = LIFETIME_YR,
    debug: bool = False,
) -> Dict[str, float]:
    """Levelize centralized + summed replacement CAPEX/OPEX on total methanol [t/a]."""
    lev = {
        "eur_t_truck_opex": levelize_opex_eur_yr_to_eur_per_t_methanol(
            replacement["opex_truck_eur_yr"], annual_methanol_t_per_yr
        ),
        "eur_t_hp_capex": levelize_meur_to_eur_per_t_methanol(
            replacement["capex_hp_meur"], annual_methanol_t_per_yr, dr, lifetime, debug=debug
        ),
        "eur_t_hp_opex": levelize_opex_meur_a_to_eur_per_t_methanol(
            replacement["opex_hp_meur"], annual_methanol_t_per_yr
        ),
        "eur_t_sort_capex": levelize_meur_to_eur_per_t_methanol(
            central["capex_sorting_meur"], annual_methanol_t_per_yr, dr, lifetime
        ),
        "eur_t_sort_opex": levelize_opex_meur_a_to_eur_per_t_methanol(
            central["opex_sorting_meur_a"], annual_methanol_t_per_yr
        ),
        "eur_t_gasif_capex": levelize_meur_to_eur_per_t_methanol(
            central["capex_gasif_meur"], annual_methanol_t_per_yr, dr, lifetime
        ),
        "eur_t_gasif_opex": levelize_opex_meur_a_to_eur_per_t_methanol(
            central["opex_gasif_meur_a"], annual_methanol_t_per_yr
        ),
        "eur_t_h2_capex": levelize_meur_to_eur_per_t_methanol(
            central["capex_h2_meur"], annual_methanol_t_per_yr, dr, lifetime
        ),
        "eur_t_h2_opex": levelize_opex_meur_a_to_eur_per_t_methanol(
            central["opex_h2_meur_a"], annual_methanol_t_per_yr
        ),
    }
    lev["eur_t_total"] = sum(lev.values())
    if debug:
        print("levelize_all_costs", lev)
    return lev


def levelize_meur_to_eur_per_t_methanol(
    capex_meur: float, annual_methanol_t_per_yr: float, dr: float, lifetime: int, debug: bool = False
) -> float:
    """Capital recovery on overnight CAPEX in MEUR → [EUR/t methanol/a].

    capex_meur: overnight CAPEX [MEUR] (1 MEUR = 1e6 EUR scalar).
    annual_methanol_t_per_yr: methanol nameplate [t/a].
    dr: discount rate [-]; lifetime: amortization period [yr].
    """
    if debug:
        print("levelize inputs", capex_meur, annual_methanol_t_per_yr, dr, lifetime)
    crf = dr * (1 + dr) ** lifetime / ((1 + dr) ** lifetime - 1)  # [-] capital recovery factor
    capex_annual_eur = capex_meur * crf * 1e6  # [EUR/yr]; MEUR is 10^6 EUR as scalar
    return capex_annual_eur / annual_methanol_t_per_yr  # [EUR/t MeOH]


def simulate_case(fuel: FuelAggregate, thermo_props: dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Run gasification → synthesis → costs for one aggregated fuel bundle.

    Return dict keys use SI tags in names where helpful: *_mj_yr, *_mwth, *_mwel, *_t_yr, *_kmol_s, *_meur, *_lev [EUR/t].
    """
    n_c = fuel.n_c_pl + fuel.n_c_bio  # [kmolC/yr] total organic C (fossil+bio)
    n_h = fuel.n_h_pl + fuel.n_h_bio  # [kmolH/yr]
    n_o = fuel.n_o_pl + fuel.n_o_bio  # [kmolO/yr]
    if n_c <= 0:
        raise ValueError("aggregate carbon moles must be positive")

    x_ratio = n_h / n_c  # [-] kmolH/kmolC dry mixed basis
    y_ratio = n_o / n_c  # [-] kmolO/kmolC

    n_steam_dry = y_ratio  # [kmolH2O/kmolC] from drying stoichiometry [R2]
    z1 = x_ratio - 2.0 * n_steam_dry  # [kmolH/kmolC] after formal drying step

    z2 = z1 / 2.0 + N_STEAM_ASSUMED  # [kmolH2/kmolC] intermediate in H2 guess
    n_h2_per_kmol_c = z2  # [kmolH2/kmolC] initial H2 in syngas guess per kmol C

    e_input = fuel.e_input_mj_yr  # [MJ/yr] dry fuel LHV sum
    e_h2 = n_h2_per_kmol_c * LHV_H2_MJ_PER_KMOL * n_c * 0.70  # [MJ/yr] H2 energy in gasified fraction
    e_co = e_input * 0.70 - e_h2  # [MJ/yr] CO energy in gasified fraction (split of 70% branch)

    n_co_per_kmol_c = (e_co / LHV_CO_MJ_PER_KMOL) / (n_c * 0.70)  # [kmolCO/kmolC]
    n_co2_per_kmol_c = 1.0 - n_co_per_kmol_c  # [kmolCO2/kmolC] closure on C mole balance per kmolC gasified
    n_o2_per_kmol_c = n_co2_per_kmol_c + (n_co_per_kmol_c - 1.0) / 2.0  # [kmolO2/kmolC] gasifier O2 (pre-WGS; fixed like gasification.py)

    n_h2_kmolc = n_h2_per_kmol_c  # [kmolH2/kmolC] updated in WGS loop
    n_co_kmolc = n_co_per_kmol_c  # [kmolCO/kmolC]
    n_co2_kmolc = n_co2_per_kmol_c  # [kmolCO2/kmolC]
    n_steam_wgs = 0.0  # [kmolH2O/kmolC] extent of shift [R4]

    e_product_mj_per_kmol_c = LHV_H2_MJ_PER_KMOL * n_h2_kmolc + LHV_CO_MJ_PER_KMOL * n_co_kmolc  # [MJ/kmolC] syngas LHV per kmolC
    while n_h2_kmolc / n_co_kmolc > 2.0:
        n_h2_kmolc -= 0.001  # [kmolH2/kmolC] per step
        n_co_kmolc += 0.001  # [kmolCO/kmolC]
        n_co2_kmolc -= 0.001  # [kmolCO2/kmolC]
        n_steam_wgs += 0.001  # [kmolH2O/kmolC]
        e_product_mj_per_kmol_c = LHV_H2_MJ_PER_KMOL * n_h2_kmolc + LHV_CO_MJ_PER_KMOL * n_co_kmolc

    q_evap = RW_EVAP_MJ_PER_KG * 18.0  # [MJ/kmolH2O] evaporation load per kmol water
    q_wgs = 43.0  # [MJ/kmol] WGS thermal term (gasification.py)
    n_steam_missing = N_STEAM_ASSUMED - n_steam_dry - n_steam_wgs  # [kmolH2O/kmolC]
    q_gasify = n_steam_missing * q_evap  # [MJ/kmolC]
    q_dry = n_steam_dry * q_evap  # [MJ/kmolC]
    q_wgs_term = n_steam_wgs * q_wgs  # [MJ/kmolC]
    q_steam_demand_mj_yr = (q_gasify + q_dry + q_wgs_term) * n_c * 0.70  # [MJ/yr] steam-related heat on gasified C

    m_combustor = (fuel.m_pl + fuel.m_bio) * 0.30 + fuel.m_ash + fuel.m_h2o  # [kg/yr] combustor residue mass
    q_residual_moisture = RW_EVAP_MJ_PER_KG * fuel.m_h2o  # [MJ/yr]
    q_steam_available = e_input * 0.30 - q_residual_moisture  # [MJ/yr] from 30% combustor energy branch

    n_o2_combustor = (fuel.n_c_pl + fuel.n_c_bio) * 0.30 / FLH_H_PER_YR / 3600.0 * AIR_RATIO_COMBUSTOR  # [kmolO2/s]

    n_c_gasified = n_c * 0.70  # [kmolC/yr] carbon sent to gasification/synthesis
    n_h2o_syngas = n_steam_wgs  # [kmolH2O/kmolC] syngas moisture (shift steam)

    n_species = {
        "H2": n_h2_kmolc * n_c_gasified,  # [kmol/yr]
        "CO": n_co_kmolc * n_c_gasified,  # [kmol/yr]
        "CO2": n_co2_kmolc * n_c_gasified,  # [kmol/yr]
        "H2O": n_h2o_syngas * n_c_gasified,  # [kmol/yr]
    }
    n_tot_syngas = sum(n_species.values())  # [kmol/yr]
    y_mix = {gas: n_flow / n_tot_syngas for gas, n_flow in n_species.items()}  # [-] mole fractions
    n_mix_s = n_tot_syngas / FLH_H_PER_YR / 3600.0  # [kmol/s] syngas molar flow at FLH

    n_co2_s = n_co2_kmolc * n_c_gasified / FLH_H_PER_YR / 3600.0  # [kmolCO2/s]
    n_h2_s = n_co2_s * 3.0  # [kmolH2/s] electrolyzer H2 for CO2 + 3H2 → … stoichiometry
    qh2_mwth = n_h2_s * LHV_H2_MJ_PER_KMOL  # [MWth] H2 LHV production rate (= MJ/s)
    ph2_mwel = qh2_mwth / ETA_ELECTROLYZER  # [MWel] electrolyzer AC power

    wcomp_mix, _, _, _ = compression_energy(
        n_mix_s,
        T1=40 + 273.15,
        P1=1.0,
        thermo_props=thermo_props,
        gas=y_mix,
        n_stages=3,
        pr=3.8,
        Tdiff=30,
        n_is=0.8,
        debug=debug,
    )
    wcomp_h2, _, _, _ = compression_energy(
        n_h2_s,
        T1=75 + 273.15,
        P1=20,
        thermo_props=thermo_props,
        gas="H2",
        n_stages=2,
        pr=1.7,
        Tdiff=60,
        n_is=0.8,
        debug=debug,
    )
    w_mix_sum = float(sum(wcomp_mix))  # [MWel] syngas compressor shaft
    w_h2_sum = float(sum(wcomp_h2))  # [MWel] electrolyzer H2 compressor shaft
    w_recycle = (w_h2_sum + w_mix_sum) * RECYCLE_RATIO  # [MWel] recycle compression proxy

    n_ch3oh = n_c_gasified  # [kmolCH3OH/yr] all gasified C → methanol (model closure)
    m_ch3oh_kg_yr = n_ch3oh * 32.0  # [kg/yr] M_CH3OH = 32 kg/kmol
    lhv_ch3oh = 21.1  # [MJ/kg]
    qch3oh_mwh_yr = m_ch3oh_kg_yr * lhv_ch3oh / 3600.0  # [MWh/yr] annual methanol LHV energy
    qch3oh_s_mwth = qch3oh_mwh_yr / FLH_H_PER_YR  # [MWth] average LHV out rate

    input_fuel_mwth = e_input / 3600.0 / FLH_H_PER_YR  # [MWth] fuel chemical power (LHV)
    q_syngas_mwth = e_product_mj_per_kmol_c * n_c * 0.70 / 3600.0 / FLH_H_PER_YR  # [MWth] CO+H2 LHV from gasified C
    power_input_mwel = ph2_mwel + w_mix_sum + w_h2_sum + w_recycle  # [MWel]
    hydrogen_produced_mwth = qh2_mwth  # [MWth]
    steam_mwth = q_steam_demand_mj_yr / 3600.0 / FLH_H_PER_YR  # [MWth] steam demand as equivalent rate

    q_synthesis_input = (
        q_syngas_mwth
        + hydrogen_produced_mwth
        + w_mix_sum
        + w_h2_sum
        + w_recycle
        + steam_mwth
    )  # [mixed] same sum as gasification.py (MWth + MWel without conversion)
    q_synthesis_output = qch3oh_s_mwth  # [MWth]
    total_input_mw = input_fuel_mwth + power_input_mwel  # [mixed] fuel MWth + electric MWel (script convention)
    ratio_synthesis = q_synthesis_output / q_synthesis_input if q_synthesis_input else float("nan")  # [-]
    eff_total = q_synthesis_output / total_input_mw if total_input_mw else float("nan")  # [-]

    n_o2_demand = n_o2_per_kmol_c * n_c_gasified / FLH_H_PER_YR / 3600.0  # [kmolO2/s] gasifier O2 feed rate

    annual_methanol_t_yr = m_ch3oh_kg_yr / 1000.0  # [t methanol/a]
    capacity_sorting_t_yr = fuel.m_tot / 1000.0  # [t waste/a] sorted waste nameplate

    capex_sorting_meur = CAPEX_SORTING_REF_MEUR * (capacity_sorting_t_yr / CAPACITY_SORTING_REF_T_PER_YR) ** SCALE_EXPONENT_K  # [MEUR]
    opex_fix_sorting_meur = capex_sorting_meur * FIXATE_CAPEX  # [MEUR/a]
    opex_var_sorting_meur = OPEX_VAR_SORTING_EUR_PER_T_WASTE * capacity_sorting_t_yr * 1e-6  # [MEUR/a]
    opex_sorting_meur_a = opex_fix_sorting_meur + opex_var_sorting_meur  # [MEUR/a]

    capex_gasif_meur = CAPEX_GASIFICATION_REF_MEUR * (annual_methanol_t_yr / CAPACITY_GASIFICATION_REF_T_PER_YR) ** SCALE_EXPONENT_K  # [MEUR]
    opex_fix_gasif_meur = capex_gasif_meur * FIXATE_CAPEX  # [MEUR/a]
    opex_var_gasif_meur = (
        OPEX_VAR_GASIFICATION_EUR_PER_MWH_FUEL * (fuel.lhv_tot_blend * fuel.m_tot) / 3600.0 * 1e-6
    )  # [MEUR/a]
    opex_energy_gasif_meur = (w_mix_sum + w_h2_sum + w_recycle) * FLH_H_PER_YR * CELC_EUR_PER_MWH * 1e-6  # [MEUR/a]
    opex_gasif_meur_a = opex_fix_gasif_meur + opex_var_gasif_meur + opex_energy_gasif_meur  # [MEUR/a]

    capex_h2_meur = CAPEX_H2_REF_KEUR_PER_MWE * ph2_mwel * 1e-3  # [MEUR]
    opex_fix_h2_meur = capex_h2_meur * FIXATE_CAPEX  # [MEUR/a]
    opex_var_h2_meur = ph2_mwel * FLH_H_PER_YR * CELC_EUR_PER_MWH * 1e-6  # [MEUR/a]
    opex_h2_meur_a = opex_fix_h2_meur + opex_var_h2_meur  # [MEUR/a]

    combusted_t_yr = m_combustor / 1000.0  # [t/a]
    added_h2_t_yr = 2.0 * hydrogen_produced_mwth / (LHV_H2_MJ_PER_KMOL / 3600.0) * FLH_H_PER_YR / 1000.0  # [t/a] electrolytic H2 mass
    consumed_o2_t_yr = n_o2_demand * 32.0 * 3600.0 * FLH_H_PER_YR / 1000.0  # [t/a] O2 (M = 32 kg/kmol)
    waste_t_yr = capacity_sorting_t_yr  # [t/a]
    methanol_t_yr = annual_methanol_t_yr  # [t/a]
    added_steam_t_yr = n_steam_missing * n_c * 0.70 * 18.0 / 1000.0  # [t/a] H2O (M = 18 kg/kmol) gasification steam make-up
    mass_diff_t_yr = (
        waste_t_yr + added_h2_t_yr + consumed_o2_t_yr + added_steam_t_yr - methanol_t_yr - combusted_t_yr
    )  # [t/a] simple mass closure residual

    # Returned scalars: MJ/yr, MWth, MWel, t/a, kmol/s, MEUR, [-], EUR/t MeOH (see keys).
    return {
        "e_input_mj_yr": e_input,  # [MJ/yr]
        "q_steam_demand_mj_yr": q_steam_demand_mj_yr,  # [MJ/yr]
        "q_steam_available_mj_yr": q_steam_available,  # [MJ/yr]
        "ratio_steam_demand_available": q_steam_demand_mj_yr / q_steam_available if q_steam_available else float("nan"),  # [-]
        "input_fuel_mwth": input_fuel_mwth,  # [MWth]
        "q_syngas_mwth": q_syngas_mwth,  # [MWth]
        "hydrogen_produced_mwth": hydrogen_produced_mwth,  # [MWth]
        "steam_mwth": steam_mwth,  # [MWth]
        "methanol_out_mwth": q_synthesis_output,  # [MWth]
        "ph2_mwel": ph2_mwel,  # [MWel]
        "wcomp_mix_mwel": w_mix_sum,  # [MWel]
        "wcomp_h2_mwel": w_h2_sum,  # [MWel]
        "wcomp_recycle_mwel": w_recycle,  # [MWel]
        "power_input_mwel": power_input_mwel,  # [MWel]
        "eta_synthesis": ratio_synthesis,  # [-] Q_meoh / Q_in (mixed units in Q_in)
        "eta_total": eff_total,  # [-] Q_meoh / (fuel_th + power_el)
        "n_o2_demand_kmol_s": n_o2_demand,  # [kmolO2/s]
        "n_o2_electrolyzer_kmol_s": n_h2_s / 2.0,  # [kmolO2/s]
        "n_o2_combustor_kmol_s": n_o2_combustor,  # [kmolO2/s]
        "waste_t_yr": waste_t_yr,  # [t/a]
        "methanol_t_yr": methanol_t_yr,  # [t/a]
        "combusted_t_yr": combusted_t_yr,  # [t/a]
        "added_h2_t_yr": added_h2_t_yr,  # [t/a]
        "consumed_o2_t_yr": consumed_o2_t_yr,  # [t/a]
        "gasif_steam_t_yr": added_steam_t_yr,  # [t/a]
        "mass_diff_t_yr": mass_diff_t_yr,  # [t/a]
        "annual_methanol_t_yr": annual_methanol_t_yr,  # [t/a]
        "capex_sorting_meur": capex_sorting_meur,  # [MEUR] centralized hub
        "capex_gasif_meur": capex_gasif_meur,  # [MEUR]
        "capex_h2_meur": capex_h2_meur,  # [MEUR]
        "opex_sorting_meur_a": opex_sorting_meur_a,  # [MEUR/a]
        "opex_gasif_meur_a": opex_gasif_meur_a,  # [MEUR/a]
        "opex_h2_meur_a": opex_h2_meur_a,  # [MEUR/a]
    }


def _fmt(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_rows", None):
        return df.to_string(index=False, float_format=lambda v: float_fmt.format(v) if pd.notna(v) else "")


def _capital_recovery_factor(dr: float, lifetime: int) -> float:
    return dr * (1 + dr) ** lifetime / ((1 + dr) ** lifetime - 1)  # [-]


def plot_final_case_cost_breakdown(
    replacement: Dict[str, Any],
    central_costs: Dict[str, float],
    lev: Dict[str, float],
    annual_methanol_t_per_yr: float,
    dr: float = DISCOUNT_RATE,
    lifetime: int = LIFETIME_YR,
    out_path: str = "results/gasify_all_final_cost_breakdown.png",
    debug: bool = False,
):
    """Three-panel figure: per-plant replacement, centralized hub, levelized EUR/t MeOH (10-plant case)."""
    per_plant = replacement["per_plant"]
    plant_names = [p["plant_name"] for p in per_plant]
    n = len(plant_names)
    x = np.arange(n)
    width = 0.26

    truck_opex_meur_a = [p["opex_truck_eur_yr"] * 1e-6 for p in per_plant]  # [MEUR/a]
    hp_opex_meur_a = [p["opex_hp_meur"] for p in per_plant]  # [MEUR/a]
    hp_capex_meur = [p["capex_hp_meur"] for p in per_plant]  # [MEUR] overnight

    fig, (ax_dec, ax_cen, ax_lev) = plt.subplots(3, 1, figsize=(14, 14))

    # Panel A — decentralized replacement costs per plant
    b1 = ax_dec.bar(x - width, truck_opex_meur_a, width, label="Truck OPEX", color="#66CCEE", edgecolor="black", linewidth=0.5)
    b2 = ax_dec.bar(x, hp_opex_meur_a, width, label="Heat pump OPEX", color="#EE6677", edgecolor="black", linewidth=0.5)
    b3 = ax_dec.bar(x + width, hp_capex_meur, width, label="Heat pump CAPEX (overnight)", color="#AA3377", edgecolor="black", linewidth=0.5)
    ax_dec.set_ylabel("Cost (MEUR/a or MEUR overnight)", fontsize=13)
    ax_dec.set_title("Decentralized replacement costs (per replaced plant)", fontsize=14)
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels(plant_names, rotation=35, ha="right", fontsize=10)
    ax_dec.legend(loc="upper right", fontsize=10)
    ax_dec.grid(axis="y", alpha=0.35)
    ax_dec.tick_params(axis="y", labelsize=11)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_dec.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    for bar in b3:
        h = bar.get_height()
        if h > 0:
            ax_dec.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Panel B — centralized gasifier (single train)
    cen_labels = ["Sorting", "Gasification", "H2 electrolyzer"]
    cen_x = np.arange(3)
    cen_w = 0.35
    capex_cen = [
        central_costs["capex_sorting_meur"],
        central_costs["capex_gasif_meur"],
        central_costs["capex_h2_meur"],
    ]
    opex_cen = [
        central_costs["opex_sorting_meur_a"],
        central_costs["opex_gasif_meur_a"],
        central_costs["opex_h2_meur_a"],
    ]
    bc1 = ax_cen.bar(cen_x - cen_w / 2, capex_cen, cen_w, label="CAPEX (overnight MEUR)", color="#4477AA", edgecolor="black", linewidth=0.5)
    bc2 = ax_cen.bar(cen_x + cen_w / 2, opex_cen, cen_w, label="OPEX (MEUR/a)", color="#228833", edgecolor="black", linewidth=0.5)
    ax_cen.set_ylabel("Cost (MEUR or MEUR/a)", fontsize=13)
    ax_cen.set_title("Centralized gasifier hub (aggregated waste from all 10 plants)", fontsize=14)
    ax_cen.set_xticks(cen_x)
    ax_cen.set_xticklabels(cen_labels, fontsize=12)
    ax_cen.legend(loc="upper right", fontsize=11)
    ax_cen.grid(axis="y", alpha=0.35)
    ax_cen.tick_params(axis="y", labelsize=11)
    for bars in (bc1, bc2):
        for bar in bars:
            h = bar.get_height()
            ax_cen.annotate(
                f"{h:.0f}" if h >= 10 else f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Panel C — levelized cost on methanol [EUR/t]
    lev_labels = [
        "Truck\nOPEX",
        "HP\nCAPEX",
        "HP\nOPEX",
        "Sort\nCAPEX",
        "Sort\nOPEX",
        "Gasif.\nCAPEX",
        "Gasif.\nOPEX",
        "H2\nCAPEX",
        "H2\nOPEX",
    ]
    lev_vals = [
        lev["eur_t_truck_opex"],
        lev["eur_t_hp_capex"],
        lev["eur_t_hp_opex"],
        lev["eur_t_sort_capex"],
        lev["eur_t_sort_opex"],
        lev["eur_t_gasif_capex"],
        lev["eur_t_gasif_opex"],
        lev["eur_t_h2_capex"],
        lev["eur_t_h2_opex"],
    ]
    lev_colors = ["#66CCEE", "#EE6677", "#EE6677", "#4477AA", "#4477AA", "#228833", "#228833", "#CCBB44", "#CCBB44"]
    bl_x = range(len(lev_vals))
    bl = ax_lev.bar(bl_x, lev_vals, color=lev_colors, edgecolor="black", linewidth=0.6)
    ax_lev.axhline(lev["eur_t_total"], color="gray", linestyle="--", linewidth=1.2, label=f"Total = {lev['eur_t_total']:.0f} EUR/t")
    ax_lev.set_xticks(list(bl_x))
    ax_lev.set_xticklabels(lev_labels, fontsize=11)
    ax_lev.set_ylabel("Levelized cost (EUR/t methanol)", fontsize=13)
    ax_lev.set_title(
        f"Levelized cost breakdown ({annual_methanol_t_per_yr:,.0f} t methanol/a)",
        fontsize=14,
    )
    ax_lev.legend(loc="upper right", fontsize=11)
    ax_lev.grid(axis="y", alpha=0.35)
    ax_lev.tick_params(axis="y", labelsize=11)
    for bar in bl:
        h = bar.get_height()
        ax_lev.annotate(
            f"{h:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Cost breakdown — 10 plants replaced by centralized gasifier", fontsize=15, y=1.01)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_final_case_cost_breakdown saved:", out_path)
    plt.close(fig)


def _annotate_bars(ax, bars, fontsize: int = 9) -> None:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


def plot_final_case_energy_breakdown(
    hub_out: Dict[str, float],
    replacement: Dict[str, Any],
    plants_slice: pd.DataFrame | None = None,
    out_path: str = "results/gasify_all_final_energy_breakdown.png",
    debug: bool = False,
):
    """Two-panel energy balance (single MW axis; heat=red, power=blue)."""
    # Heat: reds; power: blues (MWth and MWel shown as MW on one axis)
    c_fuel = "#7B241C"
    c_heat_1 = "#C0392B"
    c_heat_2 = "#E74C3C"
    c_heat_3 = "#F5B7B1"
    c_pwr_1 = "#1A5276"
    c_pwr_2 = "#2E86C1"
    c_pwr_3 = "#85C1E9"

    per_plant = replacement["per_plant"]
    names = [p["plant_name"] for p in per_plant]
    n = len(names)
    x = np.arange(n)
    w = 0.13

    qfgc_by_name: Dict[str, float] = {}
    qlhv_by_name: Dict[str, float] = {}
    if plants_slice is not None:
        for _, row in plants_slice.iterrows():
            qfgc_by_name[str(row["Name"])] = float(row["Qfgc"])
            qlhv_by_name[str(row["Name"])] = float(row["Qlhv"])

    fuel_before: List[float] = []
    boiler_loss_before: List[float] = []
    for p in per_plant:
        qlhv = qlhv_by_name.get(p["plant_name"], p["fuel_mwth"])
        qfgc = qfgc_by_name.get(p["plant_name"], 0.0)
        fuel_before.append(qlhv + qfgc)  # [MWth] plot-only: Qlhv + Qfgc
        boiler_loss_before.append((1-ETA_BOILER) * qlhv)  # [MWth] plot-only boiler loss

    heat_before = [p["heat_out_mwth"] for p in per_plant]
    power_before = [p["power_out_mwel"] for p in per_plant]
    heat_after = [p["heat_deficit_mwth"] for p in per_plant]
    power_after = [p["power_deficit_mwel"] for p in per_plant]

    sum_hp_heat = sum(heat_after)  # [MW]
    sum_hp_elec = sum(p["hp_elec_mwel"] for p in per_plant)
    sum_site_pwr = sum(power_after)

    fig, (ax_dec, ax_sum) = plt.subplots(2, 1, figsize=(14, 11))

    # Panel A — per replaced plant (before vs after HP)
    b0 = ax_dec.bar(
        x - 2.5 * w,
        boiler_loss_before,
        w,
        label=f"Boiler loss (before, {100*(1-ETA_BOILER):.0f}%)",
        color=c_heat_1,
        edgecolor="black",
        linewidth=0.4,
    )
    b1 = ax_dec.bar(x - 1.5 * w, fuel_before, w, label="Fuel in (before, HHV)", color=c_fuel, edgecolor="black", linewidth=0.4)
    b2 = ax_dec.bar(x - 0.5 * w, heat_before, w, label="Heat out (before)", color=c_heat_2, edgecolor="black", linewidth=0.4)
    b3 = ax_dec.bar(
        x + 0.5 * w,
        heat_after,
        w,
        label="Heat out (after, HP)",
        color=c_heat_3,
        edgecolor="black",
        linewidth=0.4,
        hatch="///",
    )
    b4 = ax_dec.bar(x + 1.5 * w, power_before, w, label="Power out (before)", color=c_pwr_2, edgecolor="black", linewidth=0.4)
    b5 = ax_dec.bar(
        x + 2.5 * w,
        power_after,
        w,
        label="Power deficit (after, site)",
        color=c_pwr_3,
        edgecolor="black",
        linewidth=0.4,
        hatch="///",
    )
    ax_dec.set_ylabel("Capacity [MW]", fontsize=13)
    ax_dec.set_title("Replaced decentralized plants — before shutdown vs after heat pumps", fontsize=14)
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax_dec.grid(axis="y", alpha=0.35)
    ax_dec.tick_params(axis="y", labelsize=11)
    ax_dec.legend(loc="upper right", fontsize=9, ncol=2)
    for bars in (b0, b1, b2, b3, b4, b5):
        _annotate_bars(ax_dec, bars, fontsize=8)

    # Panel B — two groups: (1) Σ HP replacement  (2) centralized gasifier hub
    eta_tot = hub_out["eta_total"]
    hp_labels = ["Σ heat out\n(HP)", "Σ power deficit\n(HP)", "Σ power deficit\n(site)"]
    hp_vals = [sum_hp_heat, sum_hp_elec, sum_site_pwr]
    hp_colors = [c_heat_2, c_pwr_2, c_pwr_3]
    hp_hatch = "///"  # after HP replacement only

    hub_labels = ["Waste fuel input", "Methanol output", "Hub power input"]
    hub_vals = [hub_out["input_fuel_mwth"], hub_out["methanol_out_mwth"], hub_out["power_input_mwel"]]
    hub_colors = [c_fuel, c_heat_1, c_pwr_1]

    x_hp = np.array([0.0, 1.0, 2.0])
    x_hub = np.array([4.5, 5.5, 6.5])
    bar_w = 0.75

    bars_hp = ax_sum.bar(
        x_hp,
        hp_vals,
        bar_w,
        color=hp_colors,
        edgecolor="black",
        linewidth=0.5,
        hatch=hp_hatch,
    )
    bars_hub = ax_sum.bar(
        x_hub,
        hub_vals,
        bar_w,
        color=hub_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax_sum.axvline(3.25, color="gray", linestyle=":", linewidth=1.0)
    ax_sum.set_xticks(list(x_hp) + list(x_hub))
    ax_sum.set_xticklabels(hp_labels + hub_labels, fontsize=11)
    ax_sum.set_ylabel("Capacity [MW]", fontsize=13)
    ax_sum.set_title(
        f"Aggregated balances — η_total (hub) = {eta_tot:.3f}  |  "
        "left: HP replacement (10 sites)  |  right: centralized gasifier",
        fontsize=13,
    )
    ax_sum.grid(axis="y", alpha=0.35)
    ax_sum.tick_params(axis="y", labelsize=11)
    ax_sum.text(0.22, -0.11, "HP replacement (Σ, 10 sites)", transform=ax_sum.transAxes, ha="center", fontsize=11)
    ax_sum.text(0.78, -0.11, "Centralized gasifier hub", transform=ax_sum.transAxes, ha="center", fontsize=11)
    _annotate_bars(ax_sum, bars_hp)
    _annotate_bars(ax_sum, bars_hub)

    fig.suptitle(
        "Energy balance — 10 plants replaced (MW; thermal MWth and electrical MWel on one axis)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_final_case_energy_breakdown saved:", out_path)
    plt.close(fig)


def main() -> None:
    plants = pd.read_csv("data/plants_clean.csv")
    plants = plants.sort_values(by="m_tot", ascending=False).reset_index(drop=True)

    if len(plants) < N_CASES:
        raise ValueError(f"Need at least {N_CASES} plants in plants_clean.csv; found {len(plants)}.")

    thermo_props = get_CoolProp()  # species cp/cv vs T for compression_energy [kJ/kgK] paths in model.py
    truck_costs = pd.read_csv("data/truck_costs.csv")

    energy_rows: List[Dict[str, Any]] = []
    mass_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    final_case: Dict[str, Any] | None = None

    for n_plants in range(1, N_CASES + 1):
        subset = plants.iloc[:n_plants]
        fuel = aggregate_fuel(subset)
        out = simulate_case(fuel, thermo_props, debug=False)
        replacement = sum_replacement_costs(subset, truck_costs, debug=False)
        central_costs = {
            "capex_sorting_meur": out["capex_sorting_meur"],
            "capex_gasif_meur": out["capex_gasif_meur"],
            "capex_h2_meur": out["capex_h2_meur"],
            "opex_sorting_meur_a": out["opex_sorting_meur_a"],
            "opex_gasif_meur_a": out["opex_gasif_meur_a"],
            "opex_h2_meur_a": out["opex_h2_meur_a"],
        }
        lev = levelize_all_costs(out["annual_methanol_t_yr"], central_costs, replacement, debug=False)
        if n_plants == N_CASES:
            final_case = {
                "replacement": replacement,
                "central_costs": central_costs,
                "lev": lev,
                "hub_out": out,
                "annual_methanol_t_per_yr": out["annual_methanol_t_yr"],
            }

        energy_rows.append(
            {
                "plants": n_plants,  # [-] count of largest sites cumulated
                "fuel_mwth": out["input_fuel_mwth"],  # [MWth]
                "syngas_mwth": out["q_syngas_mwth"],  # [MWth]
                "h2_lhv_mwth": out["hydrogen_produced_mwth"],  # [MWth]
                "steam_mwth": out["steam_mwth"],  # [MWth]
                "meoh_mwth": out["methanol_out_mwth"],  # [MWth]
                "power_mwel": out["power_input_mwel"],  # [MWel]
                "pel_mwel": out["ph2_mwel"],  # [MWel]
                "eta_synth": out["eta_synthesis"],  # [-]
                "eta_tot": out["eta_total"],  # [-]
                "Qsd_Qsa": out["ratio_steam_demand_available"],  # [-] steam demand / available (thermal MJ ratio)
            }
        )
        mass_rows.append(
            {
                "plants": n_plants,  # [-]
                "waste_kt_a": out["waste_t_yr"] / 1000.0,  # [kt/a]
                "meoh_kt_a": out["methanol_t_yr"] / 1000.0,  # [kt/a]
                "combusted_kt_a": out["combusted_t_yr"] / 1000.0,  # [kt/a]
                "h2_kt_a": out["added_h2_t_yr"] / 1000.0,  # [kt/a]
                "o2_kt_a": out["consumed_o2_t_yr"] / 1000.0,  # [kt/a]
                "steam_kt_a": out["gasif_steam_t_yr"] / 1000.0,  # [kt/a]
                "mass_diff_kt_a": out["mass_diff_t_yr"] / 1000.0,  # [kt/a]
            }
        )
        cost_rows.append(
            {
                "plants": n_plants,  # [-]
                "eur_t_truck_opex": lev["eur_t_truck_opex"],  # [EUR/t MeOH] Σ plants, variable haul
                "eur_t_hp_capex": lev["eur_t_hp_capex"],  # [EUR/t MeOH] Σ plants
                "eur_t_hp_opex": lev["eur_t_hp_opex"],  # [EUR/t MeOH] Σ plants
                "eur_t_sort_capex": lev["eur_t_sort_capex"],  # [EUR/t MeOH] centralized
                "eur_t_sort_opex": lev["eur_t_sort_opex"],  # [EUR/t MeOH]
                "eur_t_gasif_capex": lev["eur_t_gasif_capex"],  # [EUR/t MeOH]
                "eur_t_gasif_opex": lev["eur_t_gasif_opex"],  # [EUR/t MeOH]
                "eur_t_h2_capex": lev["eur_t_h2_capex"],  # [EUR/t MeOH]
                "eur_t_h2_opex": lev["eur_t_h2_opex"],  # [EUR/t MeOH]
                "eur_t_total": lev["eur_t_total"],  # [EUR/t MeOH]
                "meur_truck_capex": 0.0,  # [MEUR] no truck CAPEX
                "meur_hp": replacement["capex_hp_meur"],  # [MEUR] Σ replaced plants
                "meur_sort": out["capex_sorting_meur"],  # [MEUR] centralized
                "meur_gasif": out["capex_gasif_meur"],  # [MEUR]
                "meur_h2": out["capex_h2_meur"],  # [MEUR]
                "meur_a_truck_opex": replacement["opex_truck_eur_yr"] * 1e-6,  # [MEUR/a]
                "meur_a_hp_opex": replacement["opex_hp_meur"],  # [MEUR/a]
                "meur_a_sort_opex": out["opex_sorting_meur_a"],  # [MEUR/a]
                "meur_a_gasif_opex": out["opex_gasif_meur_a"],  # [MEUR/a]
                "meur_a_h2_opex": out["opex_h2_meur_a"],  # [MEUR/a]
            }
        )

    df_e = pd.DataFrame(energy_rows)
    df_m = pd.DataFrame(mass_rows)
    df_c = pd.DataFrame(cost_rows)

    print()
    print("=" * 100)
    print("TABLE 1 — Energy balance (rates averaged over FLH_gasifier = {} h/yr)".format(int(FLH_H_PER_YR)))
    print("=" * 100)
    print(_fmt(df_e))
    print("  Units: *_mwth [MWth], *_mwel [MWel], eta_* and Qsd_Qsa [-].")
    print()

    print("=" * 100)
    print("TABLE 2 — Material balance")
    print("=" * 100)
    print(_fmt(df_m))
    print("  Units: all mass columns [kt/a] (1000 t/a); plants [-] = number of sites cumulated.")
    print()

    print("=" * 100)
    print("TABLE 3 — Levelized costs + overnight CAPEX")
    print("=" * 100)
    print(_fmt(df_c))
    print("  Units: eur_t_* [EUR/t methanol] levelized on hub methanol; truck/HP summed over replaced sites;")
    print("  sort/gasif/H2 = single centralized train; meur_* overnight CAPEX, meur_a_* annual OPEX [MEUR/a].")
    print()

    # PRINT THE eur_t_total FOR EACH CASE
    for i in range(len(df_c)):
        print(f"If replacing {i+1} plants: {df_c.iloc[i]['eur_t_total']:.2f} EUR/t METHANOL")

    if final_case is not None:
        plot_final_case_cost_breakdown(
            replacement=final_case["replacement"],
            central_costs=final_case["central_costs"],
            lev=final_case["lev"],
            annual_methanol_t_per_yr=final_case["annual_methanol_t_per_yr"],
            debug=False,
        )
        print("Saved final-case cost breakdown figure to results/gasify_all_final_cost_breakdown.png")
        plot_final_case_energy_breakdown(
            hub_out=final_case["hub_out"],
            replacement=final_case["replacement"],
            plants_slice=plants.iloc[:N_CASES],
            debug=False,
        )
        print("Saved final-case energy breakdown figure to results/gasify_all_final_energy_breakdown.png")


if __name__ == "__main__":
    main()
