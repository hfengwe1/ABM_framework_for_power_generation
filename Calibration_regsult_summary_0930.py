#%% Main script
import os
cwd = 'c:/School/SloanProject/ABM_Model/SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'
os.chdir(cwd)
#
from random import sample
# from sqlite3 import DatabaseError
import numpy as np
import pandas as pd
# from Market_Class_0620 import Market, Merge
import matplotlib.pyplot as plt

Result_dir = os.path.join(cwd,"00 Results","Calibration", "Calibration_IRR8_Optimal" + "Sep-24" ) # save results
ncp_file = "hist_capacity_installation.csv"
CP = pd.read_csv(os.path.join(Result_dir, ncp_file))   # historical capacity installation data
NG = {}
Solar = {}
Wind = {}
Total = {}

for k in range(30):
    fuel_yr_file = os.path.join(Result_dir, "ABM_fuel_agt_"+str(k)+".xlsx")
    df = pd.read_excel(fuel_yr_file)
    NG["Run"+str(k)] = df['NG'].values
    Solar["Run"+str(k)] = df['Solar'].values
    Wind["Run"+str(k)] = df['Wind'].values
    Total["Run"+str(k)] = df['Total'].values - CP["Retire"].values 

df_NG = pd.DataFrame.from_dict(NG).cumsum()
df_Solar = pd.DataFrame.from_dict(Solar).cumsum()
df_Wind = pd.DataFrame.from_dict(Wind).cumsum()
df_Total = pd.DataFrame.from_dict(Total).cumsum() + 109179.0

# NG["Hist"] = CP["NG"].cumsum().values
# Solar["Hist"] = CP["Solar"].cumsum().values
# Wind["Hist"] = CP["Wind"].cumsum().values
# df_Total["Hist"] = CP["Total"].values

#%% Figures 
# plt.style.use('_mpl-gallery')
x = np.arange(2012,2021)  # x-axis: years
df_list = [df_NG,df_Wind,df_Solar, df_Total]
tech = ["NG", "Wind","Solar", "Total"] 
for i in range(4):
    df = df_list[i]
    y1 = df.max(axis=1)
    y2 = df.min(axis=1)
    t = tech[i]
    if t == "Total":
        y3 = CP[t].values
    else:
        y3 = CP[t].cumsum().values
    # plot
    fig, ax = plt.subplots(1,1, sharex=True, figsize=(8, 3))
    # ax3.fill_between(x, y1, y2)
    # ax3.set_title('fill between y1 and y2')
    # ax3.set_xlabel('x')
    # fig.tight_layout()
    
    ax.fill_between(x, y1, y2, alpha=0.5, linewidth=1)
    ax.plot(x, y3, linewidth=2, color = 'r')
    # ax.set_title(t)
    ax.set_ylabel('Generation CApacity (MW)')
    # ax.set_xlabel('Year')
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    # ax.set(xlim=(2012, 2021), xticks=np.arange(2013, 2020),
    #     ylim=(0, 40000), yticks=np.arange(0, 40000,5000))
    plt.savefig(os.path.join(Result_dir, t + '2.tif'), bbox_inches='tight')


    # plt.show()
# %%
