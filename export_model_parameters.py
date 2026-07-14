"""Export model constants and uncertainties from controller.py to CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path("results")
UNCERTAINTIES_PATH = RESULTS_DIR / "model_uncertainties.csv"
CONSTANTS_PATH = RESULTS_DIR / "model_constants.csv"

CONSTANT_ROWS: list[dict[str, str]] = [
    {"parameter": "plot_results", "name": "Plot diagnostic outputs", "value": "False", "units": "-", "notes": ""},
    {"parameter": "n_replacement_cases", "name": "Number of replacement cases evaluated", "value": "10", "units": "count", "notes": ""},
    {"parameter": "plants_df", "name": "CHP plant inventory dataset", "value": "External dataset", "units": "-", "notes": ""},
    {"parameter": "shipping_costs", "name": "CO2 shipping cost matrix", "value": "External dataset", "units": "-", "notes": ""},
    {"parameter": "compression_costs", "name": "CO2 compression cost data", "value": "External dataset", "units": "-", "notes": ""},
    {"parameter": "thermo_props", "name": "CoolProp thermodynamic property data", "value": "External dataset", "units": "-", "notes": ""},
    {"parameter": "SEK_to_EUR", "name": "SEK to EUR exchange rate", "value": "0.091", "units": "EUR/SEK", "notes": ""},
    {"parameter": "profit", "name": "Profit markup fraction", "value": "0.1", "units": "-", "notes": ""},
    {"parameter": "CPI2015", "name": "Consumer price index (2015)", "value": "314.21", "units": "-", "notes": ""},
    {"parameter": "CPI2025", "name": "Consumer price index (2025)", "value": "417.96", "units": "-", "notes": ""},
    {"parameter": "eta_boiler", "name": "Boiler efficiency", "value": "0.85", "units": "-", "notes": ""},
    {"parameter": "capture_rate", "name": "CO2 capture rate", "value": "0.9", "units": "-", "notes": ""},
    {"parameter": "eta_is", "name": "Isentropic efficiency", "value": "0.8", "units": "-", "notes": ""},
    {"parameter": "FLH_gasifier", "name": "Gasifier full load hours", "value": "8000", "units": "h/yr", "notes": ""},
    {"parameter": "LHV_H2_MJ_PER_KMOL", "name": "Lower heating value of hydrogen", "value": "243", "units": "MJ/kmol", "notes": ""},
    {"parameter": "LHV_CO_MJ_PER_KMOL", "name": "Lower heating value of carbon monoxide", "value": "286", "units": "MJ/kmol", "notes": ""},
    {"parameter": "LHV_CH3OH", "name": "Lower heating value of methanol", "value": "21.1", "units": "MJ/kg", "notes": ""},
    {"parameter": "recycle_ratio", "name": "Gasifier recycle ratio", "value": "3", "units": "-", "notes": ""},
    {"parameter": "capex_ref_capture_keur", "name": "Reference capture CAPEX", "value": "336700", "units": "kEUR", "notes": ""},
    {"parameter": "capacity_ref_capture_kt_per_yr", "name": "Reference capture capacity", "value": "400", "units": "ktCO2/yr", "notes": ""},
    {"parameter": "capex_ref_loading_sek", "name": "Reference CO2 loading CAPEX", "value": "63000000", "units": "SEK", "notes": ""},
    {"parameter": "capacity_ref_loading_kt_per_yr", "name": "Reference CO2 loading capacity", "value": "150", "units": "ktCO2/yr", "notes": ""},
    {"parameter": "capex_ref_sorting_msek", "name": "Reference waste sorting CAPEX", "value": "650", "units": "MSEK", "notes": ""},
    {"parameter": "capacity_ref_sorting_t_waste_per_yr", "name": "Reference sorting capacity", "value": "200000", "units": "t waste/yr", "notes": ""},
    {"parameter": "capex_ref_gasification_meur", "name": "Reference gasification CAPEX", "value": "749.729639", "units": "MEUR", "notes": ""},
    {"parameter": "capacity_ref_gasification_t_methanol_per_yr", "name": "Reference gasification capacity", "value": "237000", "units": "t methanol/yr", "notes": ""},
    {"parameter": "ADJUST_CEPCI", "name": "Apply CEPCI escalation to overnight CAPEX", "value": "True", "units": "-", "notes": ""},
    {"parameter": "CEPCI_target", "name": "Target CEPCI for cost evaluation", "value": "900", "units": "-", "notes": ""},
    {"parameter": "CEPCI_reference", "name": "General CEPCI reference", "value": "600", "units": "-", "notes": ""},
    {"parameter": "CEPCI_capture_reference", "name": "Capture CAPEX CEPCI reference", "value": "900", "units": "-", "notes": ""},
    {"parameter": "CEPCI_HP_reference", "name": "Heat pump CAPEX CEPCI reference", "value": "816", "units": "-", "notes": ""},
    {"parameter": "CEPCI_sorting_ref", "name": "Sorting CAPEX CEPCI reference", "value": "900", "units": "-", "notes": ""},
    {"parameter": "CEPCI_opex_sorting_ref", "name": "Sorting OPEX CEPCI reference", "value": "816", "units": "-", "notes": ""},
    {"parameter": "CEPCI_gasification_ref", "name": "Gasification CAPEX CEPCI reference", "value": "600", "units": "-", "notes": ""},
    {"parameter": "CEPCI_opex_gasification_ref", "name": "Gasification OPEX CEPCI reference", "value": "900", "units": "-", "notes": ""},
    {"parameter": "CEPCI_h2_ref", "name": "Electrolyzer CAPEX CEPCI reference", "value": "900", "units": "-", "notes": ""},
]

UNCERTAINTY_ROWS: list[dict[str, str]] = [
    {"parameter": "EPR_products", "name": "EPR fee applies to products (not only granulates)", "type": "Categorical", "uncertainty_range": "True; False", "units": "-", "notes": ""},
    {"parameter": "EPR_fee", "name": "EPR fee level", "type": "Categorical", "uncertainty_range": "100; 200; 300; 400; 500", "units": "EUR/tpl", "notes": ""},
    {"parameter": "baseline_granulates", "name": "Baseline plastic granulate production", "type": "Real", "uncertainty_range": "1200000 - 1300000", "units": "t/yr", "notes": ""},
    {"parameter": "inc_granulates", "name": "Granulate production increase (relative)", "type": "Real", "uncertainty_range": "0 - 0.3", "units": "-", "notes": ""},
    {"parameter": "shift_granulates", "name": "Granulate production shift (relative)", "type": "Real", "uncertainty_range": "0 - 0.4", "units": "-", "notes": ""},
    {"parameter": "price_granulates", "name": "Granulate market price", "type": "Real", "uncertainty_range": "11000 - 15000", "units": "EUR/t", "notes": ""},
    {"parameter": "baseline_products", "name": "Baseline plastic products production", "type": "Real", "uncertainty_range": "700000 - 884393", "units": "t/yr", "notes": ""},
    {"parameter": "inc_products", "name": "Products production increase (relative)", "type": "Real", "uncertainty_range": "0 - 0.3", "units": "-", "notes": ""},
    {"parameter": "shift_products", "name": "Products production shift (relative)", "type": "Real", "uncertainty_range": "0 - 0.4", "units": "-", "notes": ""},
    {"parameter": "stringent_products", "name": "Stringency of products target (relative)", "type": "Real", "uncertainty_range": "0 - 1", "units": "-", "notes": ""},
    {"parameter": "price_products", "name": "Products market price", "type": "Real", "uncertainty_range": "36000 - 56000", "units": "EUR/t", "notes": ""},
    {"parameter": "COP", "name": "Coefficient of performance (heat pump)", "type": "Real", "uncertainty_range": "2.5 - 3.5", "units": "-", "notes": ""},
    {"parameter": "eta_electrolyzer", "name": "Electrolyzer efficiency", "type": "Real", "uncertainty_range": "0.675 - 0.725", "units": "-", "notes": ""},
    {"parameter": "q_reb", "name": "Reboiler specific duty", "type": "Real", "uncertainty_range": "2.7 - 3.7", "units": "MJ/kgCO2", "notes": ""},
    {"parameter": "q_hex", "name": "Heat exchanger duty", "type": "Real", "uncertainty_range": "0.62 - 0.66", "units": "MWth/MWreb", "notes": ""},
    {"parameter": "p_capture", "name": "Capture electricity intensity", "type": "Real", "uncertainty_range": "0.08 - 0.12", "units": "MWh/tCO2", "notes": ""},
    {"parameter": "p_condition", "name": "Conditioning electricity intensity", "type": "Real", "uncertainty_range": "0.3 - 0.45", "units": "MJ/kgCO2", "notes": ""},
    {"parameter": "heat_optimism", "name": "Electrolyzer waste heat recovery optimism", "type": "Real", "uncertainty_range": "0 - 1", "units": "-", "notes": ""},
    {"parameter": "gasified_carbon_fraction", "name": "Fraction of carbon routed to gasifier", "type": "Real", "uncertainty_range": "0.65 - 0.75", "units": "-", "notes": ""},
    {"parameter": "frac_combustor_pl", "name": "Plastic share of combustor carbon", "type": "Real", "uncertainty_range": "0.15 - 0.35", "units": "-", "notes": ""},
    {"parameter": "frac_energy", "name": "Share of gasified fuel energy to methanol", "type": "Real", "uncertainty_range": "0.55 - 0.65", "units": "-", "notes": ""},
    {"parameter": "celc", "name": "Electricity price", "type": "Real", "uncertainty_range": "30 - 70", "units": "EUR/MWh", "notes": ""},
    {"parameter": "cheat", "name": "Heat price (fraction of electricity price)", "type": "Real", "uncertainty_range": "0.5 - 0.95", "units": "-", "notes": ""},
    {"parameter": "pmethanol", "name": "Methanol market price", "type": "Real", "uncertainty_range": "550 - 850", "units": "EUR/t", "notes": ""},
    {"parameter": "CRC", "name": "CRC carbon price", "type": "Real", "uncertainty_range": "50 - 250", "units": "EUR/tCO2", "notes": ""},
    {"parameter": "ETS", "name": "ETS carbon price", "type": "Real", "uncertainty_range": "50 - 250", "units": "EUR/tCO2", "notes": ""},
    {"parameter": "camine", "name": "Amine makeup cost", "type": "Real", "uncertainty_range": "40 - 50", "units": "SEK/tCO2", "notes": ""},
    {"parameter": "shipping_case", "name": "CO2 shipping cost scenario", "type": "Categorical", "uncertainty_range": "optimist_0.5Mt; pessimist_0.5Mt; optimist_1Mt; pessimist_1Mt", "units": "-", "notes": ""},
    {"parameter": "storage", "name": "CO2 storage site", "type": "Categorical", "uncertainty_range": "oygarden; kalundborg", "units": "-", "notes": ""},
    {"parameter": "storage_cost", "name": "CO2 storage cost", "type": "Real", "uncertainty_range": "5 - 58", "units": "EUR/tCO2", "notes": ""},
    {"parameter": "transport_cost_factor", "name": "Transport cost multiplier (CCS and replacement haul)", "type": "Real", "uncertainty_range": "0.8 - 1.2", "units": "-", "notes": ""},
    {"parameter": "cost_correlation_truck", "name": "Truck transport cost correlation term", "type": "Real", "uncertainty_range": "0.1 - 0.2", "units": "EUR/(t·km)", "notes": ""},
    {"parameter": "cost_correlation_pipeline", "name": "Pipeline transport cost correlation term", "type": "Real", "uncertainty_range": "0.01 - 0.03", "units": "EUR/(t·km)", "notes": ""},
    {"parameter": "k", "name": "CAPEX scaling exponent", "type": "Real", "uncertainty_range": "0.62 - 0.72", "units": "-", "notes": ""},
    {"parameter": "dr", "name": "Discount rate", "type": "Real", "uncertainty_range": "0.06 - 0.09", "units": "-", "notes": ""},
    {"parameter": "t", "name": "Project lifetime", "type": "Real", "uncertainty_range": "20 - 30", "units": "yr", "notes": ""},
    {"parameter": "capex_ref_hp_keur_per_mwth", "name": "Reference heat pump CAPEX", "type": "Real", "uncertainty_range": "800 - 920", "units": "kEUR/MWth", "notes": ""},
    {"parameter": "capex_ref_h2_keur_per_mwel", "name": "Reference electrolyzer CAPEX", "type": "Real", "uncertainty_range": "410 - 690", "units": "kEUR/MWel", "notes": ""},
    {"parameter": "capex_ref_train_eur", "name": "Reference methanol train CAPEX (15 wagons x 60 t)", "type": "Real", "uncertainty_range": "8000000 - 9000000", "units": "EUR", "notes": ""},
    {"parameter": "capex_ref_synthesis_meur", "name": "Reference methanol synthesis CAPEX", "type": "Real", "uncertainty_range": "1.6 - 2", "units": "MEUR", "notes": ""},
    {"parameter": "opex_var_sorting_sek_per_t_waste", "name": "Variable sorting OPEX", "type": "Real", "uncertainty_range": "190 - 210", "units": "SEK/t waste", "notes": ""},
    {"parameter": "opex_var_gasification_eur_per_mwh_fuel", "name": "Variable gasification OPEX", "type": "Real", "uncertainty_range": "1.35 - 1.45", "units": "EUR/MWh fuel", "notes": ""},
    {"parameter": "opex_fix", "name": "Fixed OPEX as fraction of overnight CAPEX", "type": "Real", "uncertainty_range": "0.03 - 0.05", "units": "-", "notes": ""},
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], debug: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if debug:
        print(f"Wrote {len(rows)} rows to {path}")


def export_constants(path: Path = CONSTANTS_PATH, debug: bool = False) -> None:
    _write_csv(path, ["parameter", "name", "value", "units", "notes"], CONSTANT_ROWS, debug=debug)


def export_uncertainties(path: Path = UNCERTAINTIES_PATH, debug: bool = False) -> None:
    _write_csv(
        path,
        ["parameter", "name", "type", "uncertainty_range", "units", "notes"],
        UNCERTAINTY_ROWS,
        debug=debug,
    )


def main(debug: bool = False) -> None:
    export_constants(debug=debug)
    export_uncertainties(debug=debug)


if __name__ == "__main__":
    main(debug=True)
