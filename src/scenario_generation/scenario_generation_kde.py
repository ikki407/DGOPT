# -*- coding: utf=8 -*-
"""
Scenario Generation with Kernel Density Estimation

"""

############################ Import ############################
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import time
import argparse
import ConfigParser
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None

print '##################################################\n'
print 'Scenario Generation Method: Kernel Density Estimation\n'
print '##################################################\n'

# Wait for 2 seconds
time.sleep(2)

######################## ConfigParser ########################
config_file = ConfigParser.SafeConfigParser()
config_file.read('config/parameter.conf')

########################### Argparse ##########################

parser = argparse.ArgumentParser(description='Scenario generation: kde')
parser.add_argument('--scenario_num', default=216, type=int, help='# of Scenarios') 
parser.add_argument('--data', default='miyakojima_juyo.csv', type=str, help='Demand and weather data') 
parser.add_argument('--watch_years', default=5, type=int, help='Number of sampling data') 
args = parser.parse_args()



#################### Setting scenario info. ##################

one_year = 8760
watch_years = args.watch_years
watch_hours = watch_years * one_year
scenario_num = args.scenario_num


#################### Renewable function ######################
# Transform weather data to power by using production models
cutin = 3
cutrated = 13
cutoff = 25
def wind_generation(data):
    if data < cutin:
        return 0.0
    elif cutin <= data < cutrated:
        return (data - cutin)/(cutrated - cutin)
    elif cutrated <= data < cutoff:
        return 1.0
    else:
        return 0.0

def pv_generation(data):
    pv = data['solar']
    temp = data['temp']
    G = pv * 10**6 /3600.
    delta = -0.0045
    Tcell = temp + (45 - 20)/800.0 * G
    P_pv_coef = G/1000.0 * (1 + delta*(Tcell -25))
    if P_pv_coef < 0.01:
        P_pv_coef = 0.00
    return P_pv_coef


########################## Loading data ############################
orig_data = pd.read_csv('../data/{}'.format(args.data),encoding='shift-jis')


orig_data.columns = ['year','month','day','hour','solar_radiation','wind_speed','demand','temperature','price']
orig_data = orig_data.fillna(method='ffill')

data = orig_data.copy()

yen_euro = config_file.getfloat('simulation_case', 'yen_euro')



########################## demand ############################

# Obtain data from file.
m1 = orig_data['demand'].values
m2 = orig_data['price'].values
xmin, xmax = min(m1), max(m1)
ymin, ymax = min(m2), max(m2)

# Perform a kernel density estimate (KDE) on the data
x, y = np.mgrid[xmin-500:xmax+500:100j, ymin-500:ymax+500:100j]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, x.shape)



# Saving plot
fig = plt.figure(figsize=(14, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, f, rstride=1,
        cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False
    )

#ax.set_zlim(0, 0.2)
ax.zaxis.set_major_locator(plt.LinearLocator(5))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter(''))
#surf = ax.plot_surface(x, y, f, cstride=1, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=10)

#ax.plot_wireframe(x, y, f)
#ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap=cm.jet)
#plt.show()
ax.set_title('Kernel Density Function of Demand and Price', fontsize=16)
ax.set_xlabel('Demand(kW)', fontsize=14, labelpad=30.)
ax.set_ylabel('Price(Yen/MWh)', fontsize=14, labelpad=30.)
ax.set_zlabel('p(x)', fontsize=14, labelpad=3.)
fig.colorbar(surf, shrink=0.5, aspect=7, cmap=cm.jet)
plt.savefig('scenario_generation/kde_demand_price.png')

# resamling
np.random.seed(407)
demand_price_data = kernel.resample(watch_hours).T

########################## wind_speed ######################

# Obtain data from file.
m = orig_data['wind_speed'].values
xmin, xmax = min(m), max(m)

# Perform a kernel density estimate (KDE) on the data
x = np.mgrid[xmin:xmax+2:100j]
positions = x
values = m
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, x.shape)


# Saving plot
fig = plt.figure(figsize=(14, 7))
plt.plot(x, f)
plt.title('Kernel Density Function of Wind Speed', fontsize=16)
plt.xlabel('Wind Speed(m/s)', fontsize=14)
plt.ylabel('p(x)', fontsize=14,)
plt.savefig('scenario_generation/kde_wind_speed.png')

# resamling
np.random.seed(407)
wind_speed_data = kernel.resample(watch_hours).T
wind_speed_data = map(lambda x: 0.0 if x<0.0 else x, wind_speed_data[:,0])

########################## solar_radiation、temperature ######################
# Obtain data from file.
m1 = orig_data['solar_radiation'].values
m2 = orig_data['temperature'].values
xmin, xmax = min(m1), max(m1)
ymin, ymax = min(m2), max(m2)

# Perform a kernel density estimate (KDE) on the data
x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, x.shape)



# Saving plot
fig = plt.figure(figsize=(14, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, f, rstride=1,
        cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False
    )

#ax.set_zlim(0, 0.2)
ax.zaxis.set_major_locator(plt.LinearLocator(5))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter(''))
#surf = ax.plot_surface(x, y, f, cstride=1, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=10)

#ax.plot_wireframe(x, y, f)
#ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap=cm.jet)
#plt.show()
ax.set_title('Kernel Density Function of Solar irradiance and Temperature', fontsize=16)
ax.set_xlabel('Solar irradiance(MJ/m${}^2$)', fontsize=14, labelpad=30.)
ax.set_ylabel('Temperature($^\circ$C)', fontsize=14, labelpad=30.)
ax.set_zlabel('p(x)', fontsize=14, labelpad=3.)
fig.colorbar(surf, shrink=0.5, aspect=7, cmap=cm.jet)
plt.savefig('scenario_generation/kde_solar_temp.png')

# resamling
np.random.seed(407)
solar_temp_data = kernel.resample(watch_hours).T
solar_temp_data[:,0] = map(lambda x: 0.0 if x<0.0 else x, solar_temp_data[:,0])



########################### Scenario Reduction ##########################

all_data = pd.DataFrame()

all_data['demand'] = demand_price_data[:, 0]
all_data['Price'] = demand_price_data[:, 1]
all_data['wind'] = wind_speed_data
all_data['solar'] = solar_temp_data[:, 0]
all_data['temp'] = solar_temp_data[:, 1]


# Preprocessing
all_data['Price'] /= yen_euro

# Clustering
kmeans = KMeans(n_clusters=scenario_num, random_state=407)
all_data_normalized = all_data / all_data.max()
#all_data_normalized = (all_data - all_data.mean())/ all_data.std()
all_data_normalized = all_data_normalized[['demand', 'wind', 'solar']]
kmeans.fit(all_data_normalized)

# Transform claster labels into DataFrame
all_data['cluster'] = kmeans.labels_

# Time block
cluster_time = all_data.groupby(['cluster'])['demand'].count() / float(watch_hours / one_year)
print 'sum of time blocks: {} hours'.format(cluster_time.sum())
all_data['Hours'] = all_data['cluster'].map(cluster_time)

# Probability of Time block
all_data['prob'] = 1.0


########################### Create Scenario #########################
scenario_data = all_data.groupby('cluster')[['Hours', 'Price', 'demand', 'wind', 'solar', 'temp']].mean()
scenario_data.columns = ['Hours', 'Price', 'demand', 'wind', 'solar', 'temp']

########################### Create Factor #########################

# Demand
scenario_data['Demand_f'] = scenario_data['demand'] / scenario_data['demand'].max()

# Wind
scenario_data['Wind_f'] = scenario_data['wind'].apply(wind_generation)

# PV
scenario_data['PV_f'] = scenario_data[['solar', 'temp']].apply(pv_generation, 1)

scenario_data.to_csv('scenario_generation/table_{}_prob.csv'.format(args.data.split('.')[0]), header=True)



####################### Create Scenario Dictionary #######################
Scenario_number = 1
All_scenario = {}
for idx, tmp_data in scenario_data.iterrows():

    cost_SS = tmp_data['Price']
    hours = tmp_data['Hours']
    factor = {'ld': tmp_data['Demand_f'], 'WD': tmp_data['Wind_f'], 'PV': tmp_data['PV_f']}
    prob = 1.0
    All_scenario['Scenario{}'.format(Scenario_number)] = {'cost_SS': cost_SS,
            'factor': factor,
            'hours': hours,
            'prob': prob
            }

    Scenario_number += 1

# Saving
with open('scenario_generation/scenario_list_all.pickle', 'wb') as handle:
    pickle.dump(All_scenario, handle)

print 'Created scenario file in scenario_generation/scenario_list_all.pickle'
print '===========================================\n'
