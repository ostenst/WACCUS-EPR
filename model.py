# This is the model.py file
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    Pcondition = 0.37 * mcaptured                   #[MW] [Kumar, 2023]

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

    # estimate CAPEX and construct a bid of CAPEX+transport+storage
    CAPEX_CC = 4121.7 * Vfluegas ** 0.6498          #[kEUR]
    CAPEX_CL = 7004.6 * mcaptured ** 0.5243         #[kEUR] NOTE only compression cost
    print(Vfluegas)
    print(CAPEX_CC)
    print(CAPEX_CL)
    CAPEX = x["FOAK"] * (CAPEX_CC + CAPEX_CL)
    print(CAPEX)

    CRF = (x["dr"] * (1 + x["dr"]) ** x["t"]) / ((1 + x["dr"]) ** x["t"] - 1)
    annualized_CAPEX = CAPEX * CRF
    levelized_CAPEX = annualized_CAPEX / (mcaptured/1000*3600 * x["FLH"])
    print(levelized_CAPEX*1000)

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
    FOAK = 2.20,        #[-] [Beiron, 2024] applies to CO2 capture and conditioning 
    k = 0.6857,         #[-] [Stenström, 2025]

    RENOVA = ["FOAK", "NOAK"],
    HANDELO = ["FOAK", "NOAK"], # etc. ... matters most!

    # levers 
    tax = [800,1600,2400,3200],
    groups = [0,1,2,3],

    # constants
    plants = None,
    case = ["CCUS"]         #["CCUS", "CCU", "CCS"] conduct analysis separately for the three cases
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
        "k": k,
        "tax": tax,
        "groups": groups
    }

    # RQ1: tax revenues

    # RQ2: subsidy costs
    if case == "CCUS":
        CCU_names = ["Renova"]
        CCS_names = ["Händelöverket"]
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
    print("Celsio waste-CCS (FOV) has reported CAPEX of 3715 MNOK @400 ktCO2/yr \n -> Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt\n")
    print("Use this^ to estimate CAPEX (includes C&L) - then add energy OPEX etc.!")
    
    # reading data from the present study and running EPR function
    plants = pd.read_csv("data/plants.csv")
    output = WACCUS_EPR(plants=plants, case="CCUS")

    print("\n---- Conclusions ----")
