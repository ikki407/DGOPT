# -*- coding: UTF-8 -*-
"""
Scenario Generation with Duration Curve

"""

############################ Import ############################
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle
import time
import argparse
import collections
import itertools
import numpy as np
import ConfigParser

pd.options.mode.chained_assignment = None

print '##################################################\n'
print 'Scenario Generation Method: Duration Curve\n'
print '##################################################\n'

# Wait for 2 seconds
time.sleep(2)

######################## ConfigParser ########################
config_file = ConfigParser.SafeConfigParser()
config_file.read('config/parameter.conf')

########################### Argparse ##########################

parser = argparse.ArgumentParser(description='Scenario generation')
parser.add_argument('--step1', default=2, type=int, help='# Cluster in step1')
parser.add_argument('--step2', default=4, type=int, help='# Cluster in step2')
parser.add_argument('--step3', default=3, type=int, help='# Cluster in step3')
parser.add_argument('--data', default='miyakojima_juyo.csv', type=str, help='Demand and weather data') 
args = parser.parse_args()


##################### Number of the clusters in each step #################

if args.step1 == 2:
    # Step1: Divide data to Summer and Winter (Season block)
    summer_period = [4,5,6,7,8,9]
    winter_period = [1,2,3,10,11,12]

    season_block_list = {'summer': summer_period, 'winter': winter_period}
    season_name = ['summer', 'winter']
    season_block = len(season_block_list)

elif args.step1 == 4:
    # Step1: Divide data to four seasons
    spring_period = [3,4,5]
    summer_period = [6,7,8]
    autumn_period = [9,10,11]
    winter_period = [12,1,2]

    season_block_list = {'spring': spring_period, 'summer': summer_period, 'autumn': autumn_period, 'winter': winter_period}
    season_name = ['spring', 'summer', 'autumn', 'winter']
    season_block = len(season_block_list)

# Step2: Number of clusters applying data devided in Step1 (Time Block)
cluster_step2 = args.step2

# Step3: Number of clusters applying data devided in Step2 (Scenarios)
cluster_step3 = args.step3


creating_scenarios = season_block * cluster_step2 * cluster_step3**3
print 'Creating {} scenarios ({}*{}*{}^3)'.format(creating_scenarios, season_block, cluster_step2, cluster_step3)
time.sleep(3)


############################ Data ############################
orig_data = pd.read_csv('../data/{}'.format(args.data),encoding='shift-jis')

orig_data.columns = ['year','month','day','hour','solar_radiation','wind_speed','demand','temperature','price']
orig_data = orig_data.fillna(method='ffill')

data = orig_data.copy()

yen_euro = config_file.getfloat('simulation_case', 'yen_euro')

#正規化
data_max = data[['solar_radiation','wind_speed','demand','temperature']].max()
data_min = data[['solar_radiation','wind_speed','demand','temperature']].min()

#(x - min) / (max - min)
#data[['solar_radiation','wind_speed','demand','temperature']] = (data[['solar_radiation','wind_speed','demand','temperature']] - data_min) / (data_max - data_min)

#x / max
data[['solar_radiation','wind_speed','demand','temperature']] = data[['solar_radiation','wind_speed','demand','temperature']]/ (data_max)
data[['price']] = data[['price']]/yen_euro

####################### Step1. Divide to season ######################
summer_data = data[data['month'].isin(summer_period)]
winter_data = data[data['month'].isin(winter_period)]

# sort by demand
summer_data.sort_values('demand', ascending=False, inplace=True)
winter_data.sort_values('demand', ascending=False, inplace=True)


####################### Step2. Create Time block ################
block_num = 0
peak_time = 150
for season_data in [summer_data, winter_data]:
    middle_time = (len(season_data) - peak_time*2) / 2
    time_blocks = [peak_time, middle_time, middle_time, peak_time]
    time_blocks_season = []
    for i in time_blocks:
        for j in range(i):
            time_blocks_season.append(block_num)
        block_num += 1
    season_data['time_block'] = time_blocks_season


############ Step3. Create scenarios per time block #############

def factorize(fact_data, sort_value, sub_data):
    fact_dict = []
    sub_fact_dict = []
    fact_data = fact_data.sort_values(sort_value, ascending=False)
    len_fact_data = len(fact_data)
    for i in range(3):
        if i==0:
            fact_dict.append(fact_data[0:int(0.3333*len_fact_data)][sort_value].mean())
            sub_fact_dict.append(fact_data[0:int(0.3333*len_fact_data)][sub_data].mean())
        elif i==1:
            fact_dict.append(fact_data[int(0.3333*len_fact_data):int(0.6666*len_fact_data)][sort_value].mean())
            sub_fact_dict.append(fact_data[int(0.3333*len_fact_data):int(0.6666*len_fact_data)][sub_data].mean())
        elif i==2:
            fact_dict.append(fact_data[int(0.6666*len_fact_data):][sort_value].mean())
            sub_fact_dict.append(fact_data[int(0.6666*len_fact_data):][sub_data].mean())

    return fact_dict, sub_fact_dict

# Transform weather data to power by using production models
cutin = 3
cutrated = 13
cutoff = 25
def wind_generation(data):
    data *= data_max['wind_speed']
    if data < cutin:
        return 0.0
    elif cutin <= data < cutrated:
        return (data - cutin)/(cutrated - cutin)
    elif cutrated <= data < cutoff:
        return 1.0
    else:
        return 0.0

def pv_generation(PV_data, temp_data):
    pv = PV_data*data_max['solar_radiation']
    temp = temp_data*data_max['temperature']
    G = pv * 10**6 /3600.
    delta = -0.0045
    Tcell = temp + (44 - 20)/800.0 * G
    P_pv_coef = G/1000.0 * (1 + delta*(Tcell -25))
    if P_pv_coef < 0.01:
        P_pv_coef = 0.00
    return P_pv_coef


all_scenario_data = pd.DataFrame(columns=['Blocks', 'Hours', 'Price', 'Demand_f', 'Wind_f', 'PV_f'])
block_num = 0
for season_data in[summer_data, winter_data]:
    middle_time = (len(season_data) - peak_time*2) / 2
    time_blocks = [peak_time, middle_time, middle_time, peak_time]

    for i in range(4):
        tmp = season_data[season_data.time_block.isin([block_num])]
        demand_factor, price_factor = factorize(tmp, 'demand', 'price')
        wind_factor, _ = factorize(tmp, 'wind_speed', 'demand')
        PV_factor, temp_factor = factorize(tmp,'solar_radiation', 'temperature')

        #
        price_demand = zip(price_factor, demand_factor)
        wind_factor = map(wind_generation, wind_factor)
        PV_factor = map(pv_generation, PV_factor, temp_factor)

        all_scenario_data = pd.concat([all_scenario_data, pd.DataFrame([[block_num+1]*3, [time_blocks[i]]*3, price_factor, demand_factor, wind_factor, PV_factor], index=['Blocks', 'Hours', 'Price', 'Demand_f', 'Wind_f', 'PV_f']).T])

        block_num += 1

all_scenario_data.to_csv('scenario_generation/table_{}'.format(args.data),header=True,index=None)


all_scenario_data['Demand_prob'] = 1 / 3.
all_scenario_data['Wind_prob'] = 1 / 3.
all_scenario_data['PV_prob'] = 1 / 3.

all_scenario_data.to_csv('scenario_generation/table_{}_prob.csv'.format(args.data.split('.')[0]),header=True,index=None)

####################### Create Scenario Directory #####################

Scenario_number = 1
All_scenario = {}
for i in range(1, 9):
    tmp = all_scenario_data[all_scenario_data.Blocks==i]
    for demand in range(3):
        for wind in range(3):
            for PV in range(3):
                cost_SS = tmp.ix[demand]['Price']
                hours = tmp.ix[demand]['Hours']
                factor = {'ld': tmp.ix[demand]['Demand_f'], 'WD': tmp.ix[wind]['Wind_f'], 'PV': tmp.ix[PV]['PV_f']}
                prob = 1 / 3.
                All_scenario['Scenario{}'.format(Scenario_number)] = {'cost_SS': cost_SS,
                        'factor': factor,
                        'hours': hours,
                        'prob': prob
                        }

                Scenario_number += 1



# Saving
with open('scenario_generation/scenario_list_all.pickle', 'wb') as handle:
    pickle.dump(All_scenario, handle)





# Define some random data that emulates your indeded code:
NCURVES = season_block*cluster_step2 + 2
np.random.seed(101)
curves = [np.random.random(30) for i in range(NCURVES)]
values = range(NCURVES)

fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
#jet = colors.Colormap('jet')
jet = cm = plt.get_cmap('Paired')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

def create_rainbow():
    rainbow = [ax._get_lines.color_cycle.next()]
    while True:
        nextval = ax._get_lines.color_cycle.next()
        if nextval not in rainbow:
            rainbow.append(nextval)
        else:
            return rainbow

def next_color(axis_handle=ax):
    rainbow = create_rainbow()
    double_rainbow = collections.deque(rainbow)
    nextval = ax._get_lines.color_cycle.next()
    double_rainbow.rotate(-1)
    return nextval, itertools.cycle(double_rainbow)

# 1 year
plt_legend = []
for i in season_name:
    for j in range(1,cluster_step2+1):
        plt_legend.append(i + str(j))

plot_data = pd.concat([summer_data, winter_data])
plot_data = plot_data.sort_index()
for c_num in xrange(0,season_block*cluster_step2):
    #nextval, ax._get_lines.color_cycle = next_color(ax)
    #plt.plot(plot_data.iloc[(plot_data['cluster']==c_num).values,:].index.values,plot_data.iloc[(plot_data['cluster']==c_num).values,:]['demand']*data_max['demand'])
    #print "Next color is: ", nextval
    colorVal = scalarMap.to_rgba(values[c_num])
    ax.plot(plot_data.iloc[(plot_data['time_block']==c_num).values,:].index.values,plot_data.iloc[(plot_data['time_block']==c_num).values,:]['demand'].values,color=colorVal, lw=2)
plt.legend(plt_legend, bbox_to_anchor=(1.01, 1),loc=2, fontsize=21)
plt.xlabel("Hour",fontsize=21)
plt.ylabel("Demand(p.u.)",fontsize=21)
plt.xticks(fontsize = 21)
plt.yticks(fontsize = 21)
#plt.show()
plt.savefig('scenario_generation/year_demand_duration.png')

print 'Created scenario file in scenario_generation/scenario_list_all.pickle'
print '===========================================\n'
