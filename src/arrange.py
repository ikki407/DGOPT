# -*- coding: utf-8 -*-
'''
Arrange simulation results of three cases.
'''
#============================================
#   Import
#============================================
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import pickle
import ConfigParser
import time
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

#============================================
#   Parse arguments
#============================================
parser = argparse.ArgumentParser(description='Arrange results')
parser.add_argument('--case', default='a', type=str, help='Name of simulation case')
parser.add_argument('--results_folder', default='results', type=str, help='Results directory')
args = parser.parse_args()

#============================================
#   Parse config file
#============================================
config_file = ConfigParser.SafeConfigParser()
config_file.read('config/parameter.conf')

if not os.path.exists(args.results_folder):
    os.mkdir(args.results_folder)

# Scenario data
with open('scenario_generation/scenario_list_all.pickle', 'rb') as handle:
    scenario_list_all = pickle.load(handle)

# Simulation case
case = args.case #'a','b','c'
results_folder = args.results_folder + '_' + case

print '##################################################\n'
print ' Arrange Results of Simulation Case: {} \n'.format(case)
print '##################################################\n'

# Wait for 2 seconds
time.sleep(2)

#Years
years = config_file.getint('simulation_case', 'years')

#d
discount_rate = config_file.getfloat('simulation_case', 'discount_rate')

#pwf
def pwf_init(t):
    return 1 / (1+discount_rate)**(t-1)


data = pd.read_csv('{}.csv'.format(results_folder),header=None)
data.columns = ['Stage','Node','Var','Index','value']

# Second Stage Cost
total_cost = {} 

def Compute_cost(data_):
    index = data_['Index']
    index = index.replace(' ','').split(':')
    scenario_name = data_['Node'].replace(' ','').replace('Node','')
    t = int(index[0])
    pwf_t = pwf_init(t)
    n_b_h = scenario_list_all[scenario_name]['hours']
    proba = scenario_list_all[scenario_name]['prob']

    return pwf_t*proba*n_b_h*data_['value']


closs = data.iloc[(data['Var']==' pi_loss').values,:]
total_cost['pi_loss'] = closs.apply(Compute_cost,1).sum() 

cns = data.iloc[(data['Var']==' pi_ens').values,:]
total_cost['pi_ens'] = cns.apply(Compute_cost,1).sum() 

css = data.iloc[(data['Var']==' pi_ss').values,:]
total_cost['pi_ss'] = css.apply(Compute_cost,1).sum() 

cnew = data.iloc[(data['Var']==' pi_new').values,:]
total_cost['pi_new'] = cnew.apply(Compute_cost,1).sum() 

ccb = data.iloc[(data['Var']==' pi_cb').values,:]
total_cost['pi_cb'] = ccb.apply(Compute_cost,1).sum() 

cemi = data.iloc[(data['Var']==' pi_emi').values,:]
total_cost['pi_emi'] = cemi.apply(Compute_cost,1).sum() 

com = data.iloc[(data['Var']==' pi_om').values,:]
total_cost['pi_om'] = com.apply(Compute_cost,1).sum() 

# First Stage Cost
def Compute_cost2(data_):
    index = data_['Index']
    index = index.replace(' ','')
    t = int(index)
    pwf_t = pwf_init(t)

    return pwf_t*data_['value']

investment_cost = data.iloc[(data['Var']==' InvestmentCostYear').values,:]

investment_cost = investment_cost.apply(Compute_cost2,1).sum() 

incentive_cost = data.iloc[(data['Var']==' incentive').values,:]

incentive_cost = incentive_cost.apply(Compute_cost2,1).sum()

StageCost = pd.read_csv('{}_StageCostDetail.csv'.format(results_folder), header=None)
StageCost.columns = ['Stage','Node','Scenario','Var', 'TMP', 'Value']

first = StageCost.iloc[(StageCost['Var']==' FirstStageCost').values,:]['Value'].values[0]
second = StageCost.iloc[(StageCost['Var']==' SecondStageCost').values,:]

#======================================
#   Creating DG Allocation Tables
#======================================

result = pd.read_csv('{}.csv'.format(results_folder),header=None,encoding='utf-8')

result.columns = ['Stage','Node','Var','index','value']

Y_number_of = result.iloc[(result['Var']==' X_number_of').values,:]
Y_number_of = Y_number_of.iloc[(Y_number_of['value']>0.5).values,:]
Y_number_of['value'] = np.ceil(Y_number_of['value']).values

Y_number_of['index'] = Y_number_of['index'].apply(lambda x:x.replace(' ',''))


P_WD_max = eval(config_file.get('DG', 'P_WD_max')) / 10.0**3
P_PV_max = eval(config_file.get('DG', 'P_PV_max')) / 10.0**3
Q_CB_max = eval(config_file.get('DG', 'Q_CB_max')) / 10.0**3
S_SS_max = eval(config_file.get('substation', 'S_SS_max')) / 10.0**3


kekka = {}
for i in xrange(0,years):
    #initialize
    kekka[i+1] = {'SS':[],'WD':[],'PV':[],'CB':[],'SS_POWER':[],'WIND_POWER':[],'PV_POWER':[],'CB_POWER':[]}
    WIND_POWER = 0
    PV_POWER = 0
    SS_POWER = 0
    CB_POWER = 0
    years_data = Y_number_of.iloc[(Y_number_of['index'].apply(lambda x:x.split(':')[0]).values == str(i+1)),:]
    years_data.index = range(0,len(years_data))
    for j in xrange(0,len(years_data)):
        RES = years_data.ix[j]['index'].split(':')[2].strip("\'")
        BUS = years_data.ix[j]['index'].split(':')[1]
        kekka[i+1][RES].append(BUS)
        if RES == 'PV':
            PV_POWER += years_data.ix[j]['value'] * P_PV_max
        elif RES == 'SS':
            SS_POWER += years_data.ix[j]['value'] * S_SS_max
        elif RES == 'WD':
            WIND_POWER += years_data.ix[j]['value'] * P_WD_max
        elif RES == 'CB':
            CB_POWER += years_data.ix[j]['value'] * Q_CB_max
    kekka[i+1]['PV_POWER'] = PV_POWER
    kekka[i+1]['WIND_POWER'] = WIND_POWER
    kekka[i+1]['SS_POWER'] = SS_POWER
    kekka[i+1]['CB_POWER'] = CB_POWER

print pd.DataFrame(kekka).T
final_output = pd.DataFrame(kekka).T

def split_list(data):
    a = ''
    for i in data:
        a += str(i) + ' '
    return a

final_output['PV'] = final_output['PV'].apply(lambda x: map(int,x)).apply(split_list)
final_output['WD'] = final_output['WD'].apply(lambda x: map(int,x)).apply(split_list)
final_output['SS'] = final_output['SS'].apply(lambda x: map(int,x)).apply(split_list)
final_output['CB'] = final_output['CB'].apply(lambda x: map(int,x)).apply(split_list)

final_output = final_output[['SS','WD','PV','CB','SS_POWER','WIND_POWER','PV_POWER','CB_POWER']]

# Saving Power Installed
final_output.to_csv('{}/DG_Allocation_{}.csv'.format(args.results_folder, results_folder))
# Saving Cost
index_info = pd.Index(['O&M system cost', 'Losses cost',
            'Not supplied energy cost', 'Purchased energy cost',
            'DG O&M cost', 'Capacitor bank cost', 'Emission cost', 'Total costs',\
            'O&M system cost', 'Investment costs', 'Incentive'])
cost_data = [total_cost['pi_om'],total_cost['pi_loss'],total_cost['pi_ens'],
            total_cost['pi_ss'],total_cost['pi_new'], total_cost['pi_cb'], total_cost['pi_emi'],
            first+total_cost['pi_om'], total_cost['pi_om'], investment_cost, incentive_cost]

all_cost_data = pd.DataFrame(cost_data,columns=['€'],index=index_info)
all_cost_data.to_csv('{}/Cost_{}.csv'.format(args.results_folder, results_folder))


#==============================================
# Amount of installed RES capasity in each bus
#==============================================

res_power = result.iloc[(result.Var==' X_number_of').values,:]
#
res_power['year'] = res_power['index'].apply(lambda x: int(x.split(':')[0]))
res_power['bus'] = res_power['index'].apply(lambda x: int(x.split(':')[1]))
res_power['res'] = res_power['index'].apply(lambda x: x.split(':')[2])

# WD
res_power_wd = res_power.iloc[(res_power['res']=="'WD'").values, :]
res_power_wd = res_power_wd.groupby(['year', 'bus'])['value'].sum().unstack()
res_power_wd = res_power_wd.apply(np.round)
res_power_wd = res_power_wd.iloc[:,(res_power_wd!=0).any().values]
res_power_wd *= P_WD_max
#res_power_wd.iloc[(res_power_wd!=0).any(1).values, :]

# PV
res_power_pv = res_power.iloc[(res_power['res']=="'PV'").values, :]
res_power_pv = res_power_pv.groupby(['year', 'bus'])['value'].sum().unstack()
res_power_pv = res_power_pv.apply(np.round)
res_power_pv = res_power_pv.iloc[:,(res_power_pv!=0).any().values]
res_power_pv *= P_PV_max

# SS
res_power_ss = res_power.iloc[(res_power['res']=="'SS'").values, :]
res_power_ss = res_power_ss.groupby(['year', 'bus'])['value'].sum().unstack()
res_power_ss = res_power_ss.apply(np.round)
res_power_ss = res_power_ss.iloc[:,(res_power_ss!=0).any().values]
res_power_ss *= S_SS_max

# CB
res_power_cb = res_power.iloc[(res_power['res']=="'CB'").values, :]
res_power_cb = res_power_cb.groupby(['year', 'bus'])['value'].sum().unstack()
res_power_cb = res_power_cb.apply(np.round)
res_power_cb = res_power_cb.iloc[:,(res_power_cb!=0).any().values]
res_power_cb *= Q_CB_max

# Create visualization of siting and sizing

def site_and_size(data):
    data = data[data>0.0]
    if len(data) > 0:
        return ','.join(data.index.astype(str) + '(' + data.values.astype(str) + ')')
    else:
        return ''

res_power_wd = res_power_wd.apply(site_and_size, axis=1).fillna('')
res_power_pv = res_power_pv.apply(site_and_size, axis=1).fillna('')
res_power_ss = res_power_ss.apply(site_and_size, axis=1).fillna('')
res_power_cb = res_power_cb.apply(site_and_size, axis=1).fillna('')

res_power_wd.name = 'WD'
res_power_pv.name = 'PV'
res_power_ss.name = 'SUB'
res_power_cb.name = 'CB'

siting_and_sizing = pd.concat([res_power_ss, res_power_wd, res_power_pv, res_power_cb], axis=1)
siting_and_sizing.reset_index(inplace=True)

siting_and_sizing.to_csv('{}/site_size_{}.csv'.format(args.results_folder, results_folder))
