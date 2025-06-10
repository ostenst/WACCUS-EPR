# This is the model.py file
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def estimate_CAPEX(mcaptured, x):
    """
    Estimate CAPEX for CO2 capture using power law model.
    
    Args:
        mcaptured: CO2 capture rate [kg/s]
        x: dictionary containing parameters
        
    Returns:
        CAPEX: Total CAPEX [kEUR]
        annualized_CAPEX: Annualized CAPEX [kEUR/yr]
        levelized_CAPEX: Levelized CAPEX [EUR/t]
    """
    # Convert from [kg/s] to [kt/yr]
    mannual = mcaptured/1000*3600 /1000 * x["FLH"]

    # Calculate CAPEX using power law model
    CAPEX = x["CAPEX_ref"] * (mannual / x["captured_ref"]) ** x["k"] #[kEUR]
    CAPEX *= 1 + x["FOAK"]
    
    # Calculate annualized and levelized CAPEX
    CRF = (x["dr"] * (1 + x["dr"]) ** x["t"]) / ((1 + x["dr"]) ** x["t"] - 1)
    annualized_CAPEX = CAPEX * CRF #[kEUR/yr]
    levelized_CAPEX = annualized_CAPEX / (mcaptured/1000*3600 * x["FLH"]) #[(kEUR/yr)/(t/yr)] -> [kEUR/t]
    
    return CAPEX, annualized_CAPEX, levelized_CAPEX

def plan_CCU(plant):
    CCU = 1
    bid = 2
    Ppenalty = 3
    Qpenalty = 4
    return CCU, bid, Ppenalty, Qpenalty

def plan_CCS(plant,x):
    # burn fuel
    mfuel = plant["Qwaste"] / (x["LHV"]/3600) /3600 #[kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12             #[kgCO2/s]
    Vfluegas = x["vfluegas"] * mfuel                #[Nm3/s]

    # capture and condition CO2
    mcaptured = mCO2 * 0.90                         #[kgCO2/s]
    Qreb = mcaptured * x["qreb"]                    #[MW]
    Pcapture = 0.1 * mcaptured/1000*3600            #[MW] [Beiron, 2022]
    Pcondition = 0.37 * mcaptured                   #[MW] [Kumar, 2023] incl. CO2 conditioning

    # penalize CHP
    P = plant["P"] * (1 - Qreb/plant["Qwaste"])     # assuming live steam is used for reboiler
    P = P - Pcapture - Pcondition
    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"])

    # recover heat up to 100 % of original DH - use whatever power is available for HP
    Qhex = 0.64 * Qreb                              #[MW] [Beiron, 2022]
    Qdiff = plant["Qdh"] - (Qdh + Qhex)
    if Qdiff < 0:
        raise ValueError
    else:
        Whp = Qdiff / x["COP"]
        if Whp > P: 
            Whp = P
    Qdh = Qdh + Qhex + Whp*x["COP"]
    P -= Whp
    Ppenalty = (plant["P"] - P) * plant["FLH"]      #[MWh/yr]
    Qpenalty = (plant["Qdh"] - Qdh) * plant["FLH"]  #[MWh/yr]

    # Estimate CAPEX
    CAPEX, annualized_CAPEX, levelized_CAPEX = estimate_CAPEX(mcaptured, x)
    print("CAPEX =", CAPEX) #[kEUR]
    print(levelized_CAPEX*1000) #[kEUR/t] -> [EUR/t]

    print("TODO: add transport and storage COST FUNCTION")

    FCCS = 1
    BECCS = 1
    bid = 2
    Ppenalty = 3
    Qpenalty = 4
    return FCCS, BECCS, bid, Ppenalty, Qpenalty
    
def WACCUS_EPR( 
    # uncertainties
    mpackaging = 1,
    mbuildings = 1,
    mKN39 = 884393,     #[t/a] plastic products mappable under KN39 [IVL]
    mtotal = 11000000,  #[t/a] estimated as total-incinterated (including imports) [Naturvardsverket]

    recyclable = 0.15,  #[-] fraction of simple products possible to recycle mechanically
    pproducts = 2,
    pKN39 = 46000,      #[SEK/tpl] [IVL]

    LHV = 11,           #[MJ/kgf] [Hammar]
    vfluegas = 4.7,     #[Nm3/kgf]
    Ccontent = 0.298,   #[kgC/kgf]
    fossil = 0.40,      #[-]
    qreb = 3.5,         #[MJ/kgCO2]
    COP = 3,

    FLH = 8000,
    dr = 0.075,
    t = 25,
    FOAK = 0.45/2,        #[-] [Beiron, 2024] applies to CO2 capture and conditioning 

    RENOVA = ["FOAK", "NOAK"],
    HANDELO = ["FOAK", "NOAK"], # etc. ... matters most!

    # levers 
    tax = [800,1600,2400,3200],
    groups = [0,1,2,3],

    # constants
    plants = None,
    case = ["CCUS"],         #["CCUS", "CCU", "CCS"] conduct analysis separately for the three cases
    k = 0.6857,             #[-] [Stenström, 2025]
    CAPEX_ref = 3715 * 87,  # [MNOK] -> [kEUR] NOTE: can pick other CAPEX from source:[Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]
    captured_ref = 400,     # [ktCO2/yr]
):
    x = {
        "mpackaging": mpackaging,
        "mbuildings": mbuildings,
        "mKN39": mKN39,
        "mtotal": mtotal,
        "recyclable": recyclable,
        "pproducts": pproducts,
        "pKN39": pKN39,
        "LHV": LHV,
        "vfluegas": vfluegas,
        "Ccontent": Ccontent,
        "fossil": fossil,
        "qreb": qreb,
        "COP": COP,
        "FLH": FLH,
        "dr": dr,
        "t": t,
        "FOAK": FOAK,
        "tax": tax,
        "groups": groups,
        "k": k,
        "CAPEX_ref": CAPEX_ref,
        "captured_ref": captured_ref,
    }

    # RQ1: tax revenues

    # RQ2: subsidy costs
    if case == "CCUS":
        CCU_names = ["Händelöverket"]
        CCS_names = ["Renova"]
    elif case == "CCU":
        CCU_names = ["Name1", "Name2", "Name3", "Name4", "Name5", "Name6"]
        CCS_names = None
    elif case == "CCS":
        CCU_names = None
        CCS_names = ["Name1", "Name2", "Name3", "Name4", "Name5", "Name6"]

    for _, plant in plants.iterrows():
        if plant["Name"] in CCS_names:
            FCCS, BECCS, bid, Ppenalty, Qpenalty = plan_CCS(plant, x)
        if plant["Name"] in CCU_names:
            CCU, bid, Ppenalty, Qpenalty = plan_CCU(plant)

    # RQ2: reversed auction simulation

    # RQ3: product cost increases
    # KN39, cheese, syringe, panel, cable, tire, pedal 

    output = 1
    return output

if __name__ == "__main__":
    # finding k exponent of CAPEX=CAPEX_ref⋅(mCO2/mCO2_ref)^k by linear regression: y = k * x
    df = pd.read_csv("data/capture_costs.csv")
    captured_ref = df["captured"].mean()
    CAPEX_ref = df["CAPEX"].mean()

    x = np.log(df["captured"] / captured_ref)
    y = np.log(df["CAPEX"] / CAPEX_ref)
    model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
    k = model.coef_[0]

    print(f"\nFitted model: CAPEX ≈ {CAPEX_ref:.2f} * (captured / {captured_ref:,.0f})^{k:.4f}")
    
    # Compare with Celsio case
    celsio_captured = 400  # ktCO2/yr
    celsio_reported_capex_mnok = 3715  # MNOK
    celsio_reported_capex_keur = celsio_reported_capex_mnok * 87  # Convert MNOK to kEUR
    model_predicted_capex = CAPEX_ref * (celsio_captured / captured_ref) ** k
    
    print("\nModel Validation against Celsio case:")
    print(f"Celsio waste-CCS (FOV) has reported CAPEX of {celsio_reported_capex_mnok} MNOK ({celsio_reported_capex_keur:.0f} kEUR) @{celsio_captured} ktCO2/yr")
    print(f"Model predicts: {model_predicted_capex:.2f} kEUR")
    print(f"Difference: {((model_predicted_capex - celsio_reported_capex_keur) / celsio_reported_capex_keur * 100):.1f}%")
    print(" -> Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt\n")
    print("Use this^ to estimate CAPEX (includes C&L) - then add energy OPEX etc.!")
    
    # reading data from the present study and running EPR function
    plants = pd.read_csv("data/plants.csv")
    output = WACCUS_EPR(plants=plants, k=k, case="CCUS")

    print("\n---- Conclusions ----")
