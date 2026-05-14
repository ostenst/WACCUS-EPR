"""
Cumulative gasification cases: 1 … 10 largest plants in plants_clean (by m_tot).

Each case aggregates waste streams, then runs the same thermodynamic and cost
logic as gasification.py (single virtual plant per case). Outputs three summary
tables: energy (MWth / MWel / [-]), mass (kt/a), costs (EUR/t MeOH + MEUR CAPEX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from model import compression_energy, get_CoolProp

# --- Physical / economic constants (aligned with gasification.py) ---
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

    capex_sorting = CAPEX_SORTING_REF_MEUR * (capacity_sorting_t_yr / CAPACITY_SORTING_REF_T_PER_YR) ** SCALE_EXPONENT_K  # [MEUR]
    capex_sorting_lev = levelize_meur_to_eur_per_t_methanol(
        capex_sorting, annual_methanol_t_yr, DISCOUNT_RATE, LIFETIME_YR, debug=False
    )  # [EUR/t MeOH]
    opex_fix_sorting = capex_sorting * FIXATE_CAPEX  # [MEUR/a]
    opex_var_sorting = OPEX_VAR_SORTING_EUR_PER_T_WASTE * capacity_sorting_t_yr * 1e-6  # [MEUR/a] from EUR/t × kt/a
    opex_sorting_lev = (opex_fix_sorting + opex_var_sorting) / (annual_methanol_t_yr * 1e-6)  # [EUR/t MeOH]

    capex_gasif = CAPEX_GASIFICATION_REF_MEUR * (annual_methanol_t_yr / CAPACITY_GASIFICATION_REF_T_PER_YR) ** SCALE_EXPONENT_K  # [MEUR]
    capex_gasif_lev = levelize_meur_to_eur_per_t_methanol(capex_gasif, annual_methanol_t_yr, DISCOUNT_RATE, LIFETIME_YR)  # [EUR/t MeOH]
    opex_fix_gasif = capex_gasif * FIXATE_CAPEX  # [MEUR/a]
    opex_var_gasif = (
        OPEX_VAR_GASIFICATION_EUR_PER_MWH_FUEL * (fuel.lhv_tot_blend * fuel.m_tot) / 3600.0 * 1e-6
    )  # [MEUR/a] fuel-energy–proportional non-energy OPEX
    opex_energy_gasif = (w_mix_sum + w_h2_sum + w_recycle) * FLH_H_PER_YR * CELC_EUR_PER_MWH * 1e-6  # [MEUR/a] compressor electricity
    opex_gasif_lev = (opex_fix_gasif + opex_var_gasif + opex_energy_gasif) / (annual_methanol_t_yr * 1e-6)  # [EUR/t MeOH]

    capex_h2 = CAPEX_H2_REF_KEUR_PER_MWE * ph2_mwel * 1e-3  # [MEUR] kEUR/MW × MW → MEUR
    capex_h2_lev = levelize_meur_to_eur_per_t_methanol(capex_h2, annual_methanol_t_yr, DISCOUNT_RATE, LIFETIME_YR)  # [EUR/t MeOH]
    opex_fix_h2 = capex_h2 * FIXATE_CAPEX  # [MEUR/a]
    opex_var_h2 = ph2_mwel * FLH_H_PER_YR * CELC_EUR_PER_MWH * 1e-6  # [MEUR/a] electrolyzer electricity
    opex_h2_lev = (opex_fix_h2 + opex_var_h2) / (annual_methanol_t_yr * 1e-6)  # [EUR/t MeOH]

    cost_lev = capex_sorting_lev + capex_gasif_lev + capex_h2_lev + opex_sorting_lev + opex_gasif_lev + opex_h2_lev  # [EUR/t MeOH]

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
        "capex_sorting_meur": capex_sorting,  # [MEUR] overnight
        "capex_gasif_meur": capex_gasif,  # [MEUR]
        "capex_h2_meur": capex_h2,  # [MEUR]
        "capex_sorting_lev": capex_sorting_lev,  # [EUR/t MeOH]
        "capex_gasif_lev": capex_gasif_lev,  # [EUR/t MeOH]
        "capex_h2_lev": capex_h2_lev,  # [EUR/t MeOH]
        "opex_sorting_lev": opex_sorting_lev,  # [EUR/t MeOH]
        "opex_gasif_lev": opex_gasif_lev,  # [EUR/t MeOH]
        "opex_h2_lev": opex_h2_lev,  # [EUR/t MeOH]
        "cost_lev_eur_t": cost_lev,  # [EUR/t MeOH] sum of six lev components
    }


def _fmt(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_rows", None):
        return df.to_string(index=False, float_format=lambda v: float_fmt.format(v) if pd.notna(v) else "")


def main() -> None:
    plants = pd.read_csv("data/plants_clean.csv")
    plants = plants.sort_values(by="m_tot", ascending=False).reset_index(drop=True)

    if len(plants) < N_CASES:
        raise ValueError(f"Need at least {N_CASES} plants in plants_clean.csv; found {len(plants)}.")

    thermo_props = get_CoolProp()  # species cp/cv vs T for compression_energy [kJ/kgK] paths in model.py

    energy_rows: List[Dict[str, Any]] = []
    mass_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []

    for n_plants in range(1, N_CASES + 1):
        subset = plants.iloc[:n_plants]
        fuel = aggregate_fuel(subset)
        out = simulate_case(fuel, thermo_props, debug=False)

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
                "eur_t_sort_capex": out["capex_sorting_lev"],  # [EUR/t MeOH]
                "eur_t_sort_opex": out["opex_sorting_lev"],  # [EUR/t MeOH]
                "eur_t_gasif_capex": out["capex_gasif_lev"],  # [EUR/t MeOH]
                "eur_t_gasif_opex": out["opex_gasif_lev"],  # [EUR/t MeOH]
                "eur_t_h2_capex": out["capex_h2_lev"],  # [EUR/t MeOH]
                "eur_t_h2_opex": out["opex_h2_lev"],  # [EUR/t MeOH]
                "eur_t_total": out["cost_lev_eur_t"],  # [EUR/t MeOH]
                "meur_sort": out["capex_sorting_meur"],  # [MEUR] overnight
                "meur_gasif": out["capex_gasif_meur"],  # [MEUR]
                "meur_h2": out["capex_h2_meur"],  # [MEUR]
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
    print("  Units: eur_t_* [EUR/t methanol] (annualized CAPEX+OPEX per product tonne); meur_* [MEUR] overnight blocks.")
    print()


if __name__ == "__main__":
    main()
