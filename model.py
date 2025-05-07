# This is the model.py file
import numpy as np
import pandas as pd


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
    print(CAPEX_CC)
    print(CAPEX_CL)

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

    plants = pd.read_csv("data/plants.csv")
    output = WACCUS_EPR(plants=plants, case="CCUS")

    print("\n---- Conclusions ----")
