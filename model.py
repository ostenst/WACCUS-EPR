# This is the model.py file
import numpy as np
def levelize(capex, dr=0.08, lifetime=25, capacity=None):
    """
    Calculate the levelized annual cost of a capital expenditure (CAPEX),
    optionally per unit of capacity (e.g., per ton treated).

    Parameters:
    - capex (float): Total capital cost (e.g., in $ or €).
    - dr (float): Discount rate as a decimal (e.g., 0.05 for 5%).
    - lifetime (int): Economic lifetime in years.
    - capacity (float, optional): Annual capacity/output (e.g., tons treated/year).

    Returns:
    - float: Levelized CAPEX (total or per unit of capacity if specified).
    """
    if dr == 0:
        annual_cost = capex / lifetime
    else:
        crf = (dr * (1 + dr) ** lifetime) / ((1 + dr) ** lifetime - 1)
        annual_cost = capex * crf

    if capacity:
        return annual_cost / capacity  # cost per unit
    else:
        return annual_cost  # total annualized cost
    
def WACCUS_EPR( 
    # uncertainties
    mKN39=884393,       #[t/a] plastic products mappable under KN39 [IVL]
    mtotal=11000000,    #[t/a] supplied-to-market is estimated as total-incinterated (including imports) [Naturvardsverket]
    recyclable=0.15,    #[-] fraction of simple products possible to recycle mechanically

    pKN39=46000,        #[SEK/tpl] [IVL]

    # levers 
    level_tax=2000,           #[SEK/tpl] [IVL], should be adjusted for only ~75% carbon content
    level_additional=5000,  #[SEK/tpl]
    # taxtarget=["KN39"], # implement target later


):
    # tax revenues
    msimple = mKN39 * recyclable
    mcomplex = mKN39 * (1-recyclable)
    tax_simple = (msimple + mcomplex) * level_tax #[SEK/a], ~10^9 SEK/a
    tax_complex = mcomplex * level_additional
    print(tax_simple, "Too much money for sorting facilities, probably! But makes sense to prioritize sorting etc..? Those who need more money are thermochemical measures!")
    print(tax_complex)
    print("IS TAX DETAIL LEVEL OK FOR NOW? - YES, SIMILAR TO IVL, MAYBE ADD TIME-SCENARIOS\n")

    # product cost increases
    cost_increase = (level_tax + level_additional)/pKN39 *100 #[%], assuming the complex tax level
    print(np.round(cost_increase), "% more expensive <- KN39 products")
    floor_plastic = 0.594   #[kgC/m2]
    floor_price = 500       #[SEK/m2]
    cost_increase = floor_plastic * (level_tax/(1000*0.75)) # [SEK/m2]
    cost_increase = cost_increase/floor_price *100 #[%]
    print(cost_increase, "% more expensive <- vinyl floor products")
    tire_carbon = 7.872 #[kgC/tire]
    cost_increase = tire_carbon * (level_tax/(1000*0.75)) # [SEK/tire]
    cost_increase = cost_increase/1200 *100 #[%]
    print(cost_increase, "% more expensive <- tire products")
    print("IS PRODUCTCOST DETAIL LEVEL OK FOR NOW? - NO, REDO FOR 1 SIMPLE + 1 COMPLEX PRODUCT\n")

    # subsidy costs
    capex = 350*10**6               #[SEK] [Sievert, Lund] says 350 million, so 134 million was maybe the subsidy?
    capacity = 10000*0.75           #[tC/a]
    brista = levelize(capex, capacity=capacity)
    print("presort", brista)        #[SEK/tC]

    capex = 1000*10**6              #[SEK]
    capacity = 200000*0.75          #[tC/a]
    motala = levelize(capex, capacity=capacity)
    print("aftersort", motala)      #[SEK/tC]

    capex = 16000*10**6             #[SEK]
    capacity = 650000*0.75          #[tC/a]
    borealis = levelize(capex, capacity=capacity)
    print("TCR", borealis)          #[SEK/tC]

    CCU = (10666+2666)/2            #based on Beiron
    print("CCU", CCU)               #[SEK/tC]

    CCS = 2500*3.66
    print("CCS", CCS)               #[SEK/tC]
    print("IS SUBSIDY DETAIL LEVEL OK FOR NOW? - NO, CALCULATE FOR SPECIFIC CITIES + TCR\n")

    # reversed auction simulation
    sorting_realized = tax_simple / brista # [tC/a]
    print("n bristas = ", sorting_realized/(10000*0.75))
    TCR_realized = tax_complex / borealis # [tC/a]
    print("n TCR = ", TCR_realized/(650000*0.75))
    CCU_realized = tax_complex / CCU # [tC/a]
    print("n CCU = ", CCU_realized/100000) #assuming 100ktC per site
    CCS_realized = tax_complex / CCS # [tC/a]
    print("n CCS = ", CCS_realized/100000) #assuming 100ktC per site
    print("IS AUCTION DETAIL LEVEL OK FOR NOW? - NO, IT MUST PRIORITIZE BETWEEN BIDS\n")
    print("RETHINK THE SIMPLE FUND - A KPI IS THE _remaining_ FUNDS AVAILABLE FOR COLLECTION/SORTING/RECYCLING, after ALL SORTING IS EXHAUSTED")

    output = 1
    return output

if __name__ == "__main__":

    output = WACCUS_EPR()
    print("\n---- Conclusions ----")
    print("At a fairly high additional tax for complex products, a decent amount of TCR/CCUS projects can be realized")
    print("Loads of sorting facilities can be realized - these are relatively cheap")
