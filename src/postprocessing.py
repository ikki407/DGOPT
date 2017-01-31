# -*- coding: utf-8 -*-
"""
Postprocessing script for easy-to-see
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import time

pd.options.mode.chained_assignment = None

print '##################################################\n'
print '                 Postprocessing                  \n'
print '##################################################\n'

# Wait for 2 seconds
time.sleep(1)

results_folder = 'results'
name = 'results'

# Cost
cost_a = pd.read_csv(os.path.join(results_folder,'Cost_{}_a.csv'.format(name)))
cost_b = pd.read_csv(os.path.join(results_folder,'Cost_{}_b.csv'.format(name)))
cost_c = pd.read_csv(os.path.join(results_folder,'Cost_{}_c.csv'.format(name)))

cost_a.columns = ['cost_name', '€']
cost_b.columns = ['cost_name', '€']
cost_c.columns = ['cost_name', '€']

cost_a.index = cost_a['cost_name']
cost_b.index = cost_b['cost_name']
cost_c.index = cost_c['cost_name']

del cost_a['cost_name'], cost_b['cost_name'], cost_c['cost_name']

cost_a = cost_a.iloc[~cost_a.index.duplicated()]
cost_b = cost_b.iloc[~cost_b.index.duplicated()]
cost_c = cost_c.iloc[~cost_c.index.duplicated()]

# DG allocation
alloc_a = pd.read_csv(os.path.join(results_folder,'DG_Allocation_{}_a.csv'.format(name)))
alloc_b = pd.read_csv(os.path.join(results_folder,'DG_Allocation_{}_b.csv'.format(name)))
alloc_c = pd.read_csv(os.path.join(results_folder,'DG_Allocation_{}_c.csv'.format(name)))

alloc_a.columns = [u'Years', u'SS', u'WD', u'PV', u'CB', u'SS_POWER', u'WIND_POWER', u'PV_POWER', u'CB_POWER']
alloc_b.columns = [u'Years', u'SS', u'WD', u'PV', u'CB', u'SS_POWER', u'WIND_POWER', u'PV_POWER', u'CB_POWER']
alloc_c.columns = [u'Years', u'SS', u'WD', u'PV', u'CB', u'SS_POWER', u'WIND_POWER', u'PV_POWER', u'CB_POWER']

alloc_a.index = alloc_a['Years']
alloc_b.index = alloc_b['Years']
alloc_c.index = alloc_c['Years']

del alloc_a['Years'], alloc_b['Years'], alloc_c['Years']

# Siting and sizing
site_size_a = pd.read_csv(os.path.join(results_folder,'site_size_{}_a.csv'.format(name)))
site_size_b = pd.read_csv(os.path.join(results_folder,'site_size_{}_b.csv'.format(name)))
site_size_c = pd.read_csv(os.path.join(results_folder,'site_size_{}_c.csv'.format(name)))

site_size_a.index = site_size_a['year']
site_size_b.index = site_size_b['year']
site_size_c.index = site_size_c['year']

del site_size_a['year'], site_size_b['year'], site_size_c['year']

# O&M costs
om_index = [u'Losses cost', 'Not supplied energy cost', \
            u'Purchased energy cost', u'DG O&M cost', \
            'Capacitor bank cost', 'Emission cost', u'O&M system cost']

om_cost = pd.DataFrame(columns=['a', 'b', 'c'], index=om_index)

om_cost['a'] = cost_a.ix[om_index]
om_cost['b'] = cost_b.ix[om_index]
om_cost['c'] = cost_c.ix[om_index]

# Total system costs
tsc_index = [u'O&M system cost', u'Investment costs', u'Incentive', u'Total costs']
tsc = pd.DataFrame(columns=['a', 'b', 'c'], index=tsc_index)

tsc['a'] = cost_a.ix[tsc_index]
tsc['b'] = cost_b.ix[tsc_index]
tsc['c'] = cost_c.ix[tsc_index]

# Power Installed
pi_index = alloc_a.index
pi_cols = [u'SS_POWER', u'WIND_POWER', u'PV_POWER', u'CB_POWER']

pi = pd.DataFrame(columns=pi_index.values,
                  index=pd.MultiIndex.from_product([['a','b','c'], ['SUB', 'WIND', 'PV', 'CB']]))

pi.ix['a'] = alloc_a[pi_cols].T.values
pi.ix['b'] = alloc_b[pi_cols].T.values
pi.ix['c'] = alloc_c[pi_cols].T.values

pi = pi.T
pi.index = pi_index
pi.ix['Total'] = pi.sum()

# Allocation of new DG
dg_index = alloc_a.index
dg_cols = [u'SS', u'WD', u'PV', 'CB']

dg = pd.DataFrame(columns=dg_index.values,
                  index=pd.MultiIndex.from_product([['a','b','c'], ['SUB', 'WIND', 'PV', 'CB']]))

dg.ix['a'] = alloc_a[dg_cols].T.values
dg.ix['b'] = alloc_b[dg_cols].T.values
dg.ix['c'] = alloc_c[dg_cols].T.values

dg = dg.T
dg.index = dg_index

# Siting and sizing
sisi_index = site_size_a.index
sisi_cols = [u'SUB', u'WD', u'PV', u'CB']

sisi = pd.DataFrame(columns=sisi_index.values,
                  index=pd.MultiIndex.from_product([['a','b','c'], ['SUB', 'WIND', 'PV', 'CB']]))

sisi.ix['a'] = site_size_a[sisi_cols].T.values
sisi.ix['b'] = site_size_b[sisi_cols].T.values
sisi.ix['c'] = site_size_c[sisi_cols].T.values

sisi = sisi.T
sisi.index = sisi_index


# ----- Saving to .csv -----
print 'Arrange and Saving all results in {}'.format(results_folder)

print 'O&M_Cost_summary.csv'
print 'TotalSystemCosts_summary.csv'
print 'PowerInstalled_summary.csv'
print 'DG_Allocation_summaryy.csv'
print 'Siting_Sizing_summary.csv'

om_cost.to_csv(os.path.join(results_folder,'O&M_Cost_summary.csv'), index=True)
tsc.to_csv(os.path.join(results_folder,'TotalSystemCosts_summary.csv'), index=True)
pi.to_csv(os.path.join(results_folder,'PowerInstalled_summary.csv'), index=True)
dg.to_csv(os.path.join(results_folder,'DG_Allocation_summary.csv'), index=True)
sisi.to_csv(os.path.join(results_folder,'Siting_Sizing_summary.csv'), index=True)

print 'Done'
