# -*- coding: UTF-8 -*-
"""
Scenario Generation with K-means

Data: Demand, Wind Speed, Solar Radiation, Temperature, Energy Price
      for 8760 hours (1 year)

Step1: 
Step2:
Step3:
Step4:
Step5:
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
print 'Scenario Generation Method: Kmeans \n'
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

data = orig_data

##################### Yen / Euro #####################
yen_euro = config_file.getfloat('simulation_case', 'yen_euro')

# Normalize data
data_max = data[['solar_radiation','wind_speed','demand','temperature']].max()
data_min = data[['solar_radiation','wind_speed','demand','temperature']].min()

#(x - min) / (max - min)
#data[['solar_radiation','wind_speed','demand','temperature']] = (data[['solar_radiation','wind_speed','demand','temperature']] - data_min) / (data_max - data_min)

#x / max
data[['solar_radiation','wind_speed','demand','temperature']] = data[['solar_radiation','wind_speed','demand','temperature']]/ (data_max)


############################ Step1 ############################

# Seasonal data
data_all_season = {}
for i in season_name:
    data_all_season[i] = data.iloc[(data['month'].isin(season_block_list[i])).values,:]

############################ Step2 ############################


for i in season_name:
    data_season_tmp = data_all_season[i]

    # Apply K-means only to data devided in Step1
    clf = KMeans(n_clusters=cluster_step2, random_state=407)
    clf.fit(data_season_tmp['demand'].values.reshape(-1,1))
    data_season_tmp['cluster'] = clf.labels_

    #Time Block計算
    TB_summer = data_season_tmp['cluster'].value_counts()

    data_season_tmp['time_block'] = data_season_tmp['cluster'].apply(lambda x: TB_summer[x])

    data_season_tmp = data_season_tmp[['solar_radiation', 'wind_speed', 'demand', 'temperature', 'cluster', 'time_block','price']]

    data_all_season[i] = data_season_tmp


# Plot
plot_data = pd.DataFrame()
cluster_increase_num = 0
for i in season_name:
    plot_data = pd.concat([plot_data, data_all_season[i]])
    season_index = data_all_season[i].index
    plot_data.loc[season_index,'cluster'] = plot_data.ix[season_index]['cluster'] + cluster_increase_num
    cluster_increase_num += cluster_step2

plot_data = plot_data.sort_index()

# define some random data that emulates your indeded code:
NCURVES = season_block*cluster_step2 + 2
np.random.seed(101)
curves = [np.random.random(30) for i in range(NCURVES)]
values = range(NCURVES)

jet = cm = plt.get_cmap('Paired')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
print scalarMap.get_clim()


fig = plt.figure(figsize=(46,24))
ax = fig.add_subplot(111)
# replace the next line
#jet = colors.Colormap('jet')
# with

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

#1 year
plt_legend = []
for i in season_name:
    for j in range(1,cluster_step2+1):
        plt_legend.append(i + str(j))

winter_seperate_list = [range(4380), range(4380, 8760)]
for c_num in xrange(0,season_block*cluster_step2):
    #nextval, ax._get_lines.color_cycle = next_color(ax)
    #plt.plot(plot_data.iloc[(plot_data['cluster']==c_num).values,:].index.values,plot_data.iloc[(plot_data['cluster']==c_num).values,:]['需要']*data_max['需要'])
    #print "Next color is: ", nextval
    colorVal = scalarMap.to_rgba(values[c_num])
    if c_num in range(cluster_step2*(season_block - 1), season_block*cluster_step2):
        for k in winter_seperate_list:
            
            tmp_index = plot_data.iloc[(plot_data['cluster']==c_num).values,:].index.values
            tmp_values = plot_data.iloc[(plot_data['cluster']==c_num).values,:]['demand'].values
            tmp_bool = map(lambda x: x in k, tmp_index)

            tmp_index = pd.Series(tmp_index)[tmp_bool].values
            tmp_values = pd.Series(tmp_values)[tmp_bool].values
            ax.plot(tmp_index, tmp_values, color=colorVal, lw=2)
            
    else:
        ax.plot(plot_data.iloc[(plot_data['cluster']==c_num).values,:].index.values,plot_data.iloc[(plot_data['cluster']==c_num).values,:]['demand'].values,color=colorVal, lw=2)
plt.legend(plt_legend, bbox_to_anchor=(1.01, 1),loc=2, fontsize=45)
plt.xlabel("Hour",fontsize=55)
plt.ylabel("Demand(p.u.)",fontsize=55)
plt.xticks(fontsize = 55)
plt.yticks(fontsize = 55)
#plt.show()
#plt.tight_layout()
plt.xlim(0,8760)
plt.subplots_adjust(right=0.8)
plt.savefig('scenario_generation/year_demand.png')
plt.close()


############################ Step3 - Step6 ############################
# Apply K-means to demand, wind speed, solar radiation devided in Step2

final_data = pd.DataFrame(columns=['Blocks','Hours','Price','Demand_f','Wind_f','PV_f'])
blocks = 1

prob_demand = []
prob_wind = []
prob_pv = []

for season in season_name: # Season 
    data_season = data_all_season[season]

    for cluster_num in xrange(0, cluster_step2): # Apply K-means
        data_block = data_season.iloc[(data_season['cluster']==cluster_num).values,:]
        clf = KMeans(n_clusters=cluster_step3, random_state=407)
        clf.fit(data_block['demand'].values.reshape(-1,1))
        data_block['step3_demand'] = clf.labels_
        aaa = data_block.groupby('step3_demand').mean()[['demand','price']]
        aaa = aaa.sort_values('demand',ascending=False)
        step3_demand = aaa['demand'].values
        step3_price = aaa['price'].values

        prob_demand_tmp = data_block.groupby('step3_demand').count()['demand'].values
        prob_demand_tmp = prob_demand_tmp / float(len(data_block))
        prob_demand.extend(prob_demand_tmp)


        clf = KMeans(n_clusters=cluster_step3, random_state=407)
        clf.fit(data_block['wind_speed'].values.reshape(-1,1))
        data_block['step3_wind'] = clf.labels_
        step3_wind = data_block.groupby('step3_wind').mean()['wind_speed'].values
        step3_wind = sorted(step3_wind,reverse=True)

        prob_wind_tmp = data_block.groupby('step3_wind').count()['wind_speed'].values
        prob_wind_tmp = prob_wind_tmp / float(len(data_block))
        prob_wind.extend(prob_wind_tmp)


        clf = KMeans(n_clusters=cluster_step3, random_state=407)
        clf.fit(data_block['solar_radiation'].values.reshape(-1,1))
        data_block['step3_pv'] = clf.labels_
        aaa = data_block.groupby('step3_pv').mean()[['solar_radiation','temperature']]
        aaa = aaa.sort_values('solar_radiation',ascending=False)
        step3_pv = aaa['solar_radiation'].values
        step3_temp = aaa['temperature'].values

        prob_pv_tmp = data_block.groupby('step3_pv').count()['solar_radiation'].values
        prob_pv_tmp = prob_pv_tmp / float(len(data_block))
        prob_pv.extend(prob_pv_tmp)

        # Price of Step3
        # Price = f(step3_demand)

        final_data = final_data.append(pd.DataFrame({'Blocks':[blocks]*cluster_step3,'Hours':[data_block['time_block'].values[0]]*cluster_step3,'Price':step3_price,'Demand_f':step3_demand,'Wind_f':step3_wind,'PV_f':step3_pv,'temp':step3_temp}))

        blocks += 1

final_data = final_data[['Blocks','Hours','Price','Demand_f','Wind_f','PV_f','temp']]
final_data.columns = ['Blocks','Hours','Price','Demand_f','Wind','PV','temp']


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

def pv_generation(data):
    pv = data['PV']*data_max['solar_radiation']
    temp = data['temp']*data_max['temperature']
    G = pv * 10**6 /3600.
    delta = -0.0045
    Tcell = temp + (45 - 20)/800.0 * G
    P_pv_coef = G/1000.0 * (1 + delta*(Tcell -25))
    if P_pv_coef < 0.01:
        P_pv_coef = 0.00
    return P_pv_coef

final_data['Wind_f'] = final_data['Wind'].apply(wind_generation)
final_data['PV_f'] = final_data[['PV','temp']].apply(pv_generation,1)

final_data = final_data[['Blocks','Hours','Price','Demand_f','Wind','Wind_f','PV','PV_f']]
#print final_data

output_data = final_data[['Blocks','Hours','Price','Demand_f','Wind_f','PV_f']]
output_data['Price'] = output_data['Price']/yen_euro
#print output_data


# For new probabilities
output_data_prob = output_data.copy()
output_data_prob['Demand_prob'] = prob_demand
output_data_prob['Wind_prob'] = prob_wind
output_data_prob['PV_prob'] = prob_pv
output_data_prob.to_csv('scenario_generation/table_{}_prob.csv'.format(args.data.split('.')[0]),header=True,index=None)

output_data['Demand_f'] = output_data['Demand_f'].apply(lambda x: round(x,2)).astype(str) + ' (' + map(lambda x: str(round(x, 3)), prob_demand) + ')'
output_data['Wind_f'] = output_data['Wind_f'].apply(lambda x: round(x,2)).astype(str) + ' (' + map(lambda x: str(round(x, 3)), prob_wind) + ')'
output_data['PV_f'] = output_data['PV_f'].apply(lambda x: round(x,2)).astype(str) + ' (' + map(lambda x: str(round(x, 3)), prob_pv) + ')'



output_data.to_csv('scenario_generation/table_{}(prob).csv'.format(args.data.split('.')[0]),header=True,index=None)


################################################################################
#
# Saving dictionary of scenarios for building optimization problem
#
# Example:
# {'Scenario1': {'factor' : {'PV': 0.05, 'WD': 0.65, 'ld': 0.78},
#             'cost_SS': 94.15,
#             'hours' : 420,
#             'prob' : 0.0102,
#             }
# ...
# }
# -----------------------------------------------------------------------------
# - key = Scenario Name : Scenario1, Scenario2,...
#
# - Scenario factors: key=>'factor', value=>{'PV': 0.05, 'WD': 0.65, 'ld': 0.78}
#
# - Purchased energy cost: key=>'cost_SS', value=>94.15
#
# - Number of hours of its scenario: key=>'hours', value=>420
#
# - Probability: key=>'prob', value=>0.0102
#
################################################################################

# Scenario dictionary
Scenario_table = {}

# Create scenario name in order of time block
block_list = output_data_prob.Blocks.unique()
block_list.sort()

# Initial scenario number
scenario_number = 1

for block_list_iter in block_list:
    tmp_data = output_data_prob[output_data_prob.Blocks==block_list_iter]
    tmp_hours = tmp_data.Hours.unique()[0]

    # Demand
    for demand_iter in range(cluster_step3):
        tmp_demand = tmp_data.loc[demand_iter,'Demand_f']
        tmp_price = tmp_data.loc[demand_iter,'Price']
        tmp_demand_prob = tmp_data.loc[demand_iter,'Demand_prob']

        # Wind
        for wind_iter in range(cluster_step3):
            tmp_wind = tmp_data.loc[wind_iter,'Wind_f']
            tmp_wind_prob = tmp_data.loc[wind_iter,'Wind_prob']

            # PV
            for PV_iter in range(cluster_step3):
                tmp_PV = tmp_data.loc[PV_iter,'PV_f']
                tmp_PV_prob = tmp_data.loc[wind_iter,'PV_prob']


                scenario_name = 'Scenario{}'.format(scenario_number)
                factor_data = {'PV': tmp_PV, 'WD': tmp_wind, 'ld': tmp_demand}
                cost_SS = tmp_price
                hours = tmp_hours
                prob = tmp_demand_prob * tmp_wind_prob * tmp_PV_prob

                Scenario_table[scenario_name] = {'factor': factor_data,
                        'cost_SS': cost_SS,
                        'hours': hours,
                        'prob': prob,
                        }

                scenario_number += 1


with open('scenario_generation/scenario_list_all.pickle', 'wb') as handle:
    pickle.dump(Scenario_table, handle)

print 'Created scenario file in scenario_generation/scenario_list_all.pickle'
print '===========================================\n'
