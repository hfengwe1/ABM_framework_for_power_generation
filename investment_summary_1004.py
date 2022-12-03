#%% 
import os
cwd = 'c:/School/SloanProject/ABM_Model/SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'
os.chdir(cwd)

from random import sample
import numpy as np
import pandas as pd
from ABM_function import create_dir
import matplotlib.pyplot as plt

def aggregation_fuel_future(n_agt, result):
    G_name =['NG','Wind','Solar']             # generation types
    row_list = []  
    for agt_i in range(n_agt):
        subset = result[result["Agt_I"] == agt_i]
        subset_invest = subset.iloc[:,0:6]  # the investment in Load Zones
        subset_fuel = subset.iloc[:,6:]     # the year, agent id, fuel invested in Load Zones
        
        for i in np.arange(len(subset_fuel.index)):  # loop through year:2012- 2020
            y = i + 2021
            invest_fuel = {"Year": y, "Agt_I": agt_i, 'NG': 0.0,'Wind': 0.0,'Solar': 0.0}  # initial value: [NG, Wind, Solar]            
    
            for lz in range(6):    # there are five load zones
                if subset_fuel.iloc[i, lz+2] == "NG":                 # the tech is NG
                    invest_fuel["NG"] += subset_invest.iloc[i, lz] # initial value     
                elif subset_fuel.iloc[i, lz+2] == "Wind":             # the tech is Solar       
                    invest_fuel["Wind"] +=  subset_invest.iloc[i, lz] # initial value     
                elif subset_fuel.iloc[i, lz+2] == "Solar":            # the tech is Solar       
                    invest_fuel["Solar"] +=  subset_invest.iloc[i, lz] # initial value 
                else: 
                    print("something went wrong")
            row_list.append(invest_fuel)                   

    fuel_agt = pd.DataFrame(row_list)  # create a dataframe for investment agrregated by fuel types
    fuel_agt_i = fuel_agt.groupby("Agt_I").sum()  # sum the investment by fuel types for each year
    del fuel_agt_i["Year"]                      # delete the Agt ID column
    # fuel_yr["Total"] = fuel_yr.sum(axis=1)    # add a column for the total invesetment
    # investment aggregated by Load Zones
    # subset = result.iloc[:,0:7]               # the investment in Load Zones
    subset_agg = result.iloc[:,0:7].groupby("Year").sum()  # sum the investment by fuel types for each year
    subset_yr = subset_agg.cumsum()  # calculate the cumulative sum of the investment in the LZs

    return fuel_agt_i, subset_yr
#%%
# Baseline
future_dir = "Future01008_cost3perct_RandomWalk_REC30"  # the future results directory name
# future_dir = "Future1008_cost3perct_RandomWalk_REC15"
# future_dir = "Future1002_demand_1perct_baseline"
# future_dir = "Future0930_cost5perct_RandomWalk"
# future_dir = "Future0930_cost3perct_RandomWalk"  # the future results directory name
# future_dir = "Future0930_cost3perct_RandomWalk_REC15"  # the future results directory name
# future_dir = "Future0930_cost5perct_RandomWalk_REC15"  # the future results directory name

summary_dir = "Summary"  # the future results directory name
create_dir(os.path.join(cwd,"00 Results",future_dir, summary_dir))  # create a directory to stroe result summaries. 

S = 100  # number of simulation runs for uncertainty analysis
n_agt = 161
LZs = ['LZ_AEN', 'LZ_CPS', 'LZ_HOUSTON', 'LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST'] # load zones
LZ_yr_S = pd.DataFrame()

for s in range(S):
    file = os.path.join(cwd,"00 Results",future_dir, "Result_1001_"+str(s)+".xlsx")
    result = pd.read_excel(file, header=0, index_col= 0)
    # organize investment by agent and fuel type
    fuel_agt_i, LZ_yr = aggregation_fuel_future(n_agt, result)  # fuel_agt: agents' investment in fuel types
                                                        # fuel_yr: total investment in LZs from 2021-2051
    LZ_yr.columns = [ x + '_' + str(s) for x in LZs]
    LZ_yr_S = pd.concat([LZ_yr_S, LZ_yr], axis=1)

LZ_year_file = os.path.join(cwd,"00 Results",future_dir, summary_dir, "LZ_year.xlsx")
LZ_yr_S.to_excel(LZ_year_file)
col = LZ_yr_S.columns

#%%
for lz in LZs:
    lz_col = [c for c in col if c.startswith(lz)]
    df = LZ_yr_S[lz_col]  # organized by load zone 
    file = os.path.join(cwd,"00 Results",future_dir, summary_dir, lz + ".xlsx")
    df.to_excel(file)
    df.plot(kind = "line",grid = True, legend = False, figsize = (4,5))
    plt.rcParams.update({'font.size': 12})
    plt.ylabel("Cumulative Generation Capacity Investment (MW)")
    plt.xlim([2020,2050]) # x axis range
    plt.ylim([0,60000]) # y axis range
    plt.title(lz)
    fig_file = os.path.join(cwd,"00 Results",future_dir, summary_dir, lz + ".tif")
    plt.tight_layout()
    plt.savefig(fig_file)

    # df.to_excel(file)    

# %%
