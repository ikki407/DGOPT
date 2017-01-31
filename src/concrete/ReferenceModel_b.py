#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
New Scenario-Based Stochastic Programming Problem for \
Long-Term Allocation of Renewable Distributed Generations

This script mainly uses Pyomo, which is an optimization framework including stochastic programming.
"""

#=================================
#   Import
#=================================
import warnings
warnings.filterwarnings("ignore")

from pyomo.core import *
import numpy as np
import pandas as pd
import pickle
import time
import ConfigParser

pd.options.mode.chained_assignment = None

#=================================
#   Parse config file
#=================================

general_config_file = ConfigParser.SafeConfigParser()
general_config_file.read('config/general.conf')

config_file = ConfigParser.SafeConfigParser()
config_file.read('config/parameter.conf')


#=================================
#   Simulation case
#=================================

#'a','b','c'
case = 'b' 
# Flag of reverse power constraints
reverse_case = config_file.getint('simulation_case', 'reverse_case')
# Flag of DG-DGIG
dg_dfig_vsi_mode = config_file.getboolean('simulation_case', 'dg_dfig_vsi_mode') 


print '##################################################\n'
print 'simulation case: %s \n' %case
print '##################################################\n'

# Wait for 2 seconds
time.sleep(2)

#=================================
#   Loading data
#=================================
# Scenario data
print 'Loading scenario data'
with open('scenario_generation/scenario_list_all.pickle', 'rb') as handle:
    scenario_list_all = pickle.load(handle)

# Distribution data
print 'Loading {}-bus distribution system'.format(general_config_file.getint('System', 'bus_number'))
system_data = pd.read_csv('system_data/{}-bus-data.csv'.format(general_config_file.getint('System', 'bus_number')))
branches = pd.read_csv('system_data/{}-bus-data-branches.csv'.format(general_config_file.getint('System', 'bus_number')))
branches = map(tuple, branches.values)

#=================================
#   Model
#=================================
print 'Starting Model Construction'

model = ConcreteModel()

#=================================
#   Sets
#=================================
model.SS = Set(initialize=['SS'])

model.RES = Set(initialize=['PV','WD'])

model.CB = Set(initialize=['CB'])

model.RESnNS = Set(initialize=['PV','WD','NS'])

model.RESnSS = Set(initialize=['PV','WD','SS'])

model.RESnSSnCB = Set(initialize=['PV','WD','SS','CB'])

model.RESnCB = Set(initialize=['PV','WD','CB'])

model.RESnld = Set(initialize=['PV','WD','ld'])

model.RESnSSnNS = Set(initialize=['PV','WD','SS','NS'])

model.LoadBuses = Set(initialize=range(1,len(system_data)+1))

model.Branches = Set(initialize=branches)

model.SSBuses = Set(initialize=[1])


#=================================
#   Parameters
#=================================

#Years
years = config_file.getint('simulation_case', 'years')
model.Years = Set(initialize=range(1,(years+1)))

#X_PV_n_max, X_WD_n_max
model.X_PV_n_max = Param(initialize=config_file.getint('DG', 'X_PV_n_max'))
model.X_WD_n_max = Param(initialize=config_file.getint('DG', 'X_WD_n_max'))
model.X_CB_n_max = Param(initialize=config_file.getint('DG', 'X_CB_n_max'))

# Power
#Base power, voltage, current, and impedance
model.S_base = Param(initialize=eval(config_file.get('system_base', 'S_base')))
model.V_base = Param(initialize=eval(config_file.get('system_base', 'V_base')))

model.I_base = Param(initialize=model.S_base / (model.V_base * np.sqrt(3)))
#model.I_base = Param(initialize=model.S_base / (model.V_base))

model.Z_base = Param(initialize=model.V_base * model.V_base / model.S_base)

substation_base_volt = eval(config_file.get('system_base', 'substation_base_volt'))

# Reverse power flow limits MW
model.reverse_limit = Param(initialize=eval(config_file.get('reverse_power_flow', 'reverse_limit')) / model.S_base)

model.Q_CB_max = Param(initialize=eval(config_file.get('DG', 'Q_CB_max'))  / (model.S_base)) 

# Investment cost of substation, DG, and CB
pi_inv_dic = {
        'PV':config_file.getfloat('investment_cost', 'inv_PV'), 
        'WD':config_file.getfloat('investment_cost', 'inv_WD'), 
        'SS':config_file.getfloat('investment_cost', 'inv_SS'),
        'CB':config_file.getfloat('investment_cost', 'inv_CB')*model.Q_CB_max.value*model.S_base
}

model.L = Param(initialize=config_file.getint('DG', 'L'))
model.interest_rate = Param(initialize=config_file.getfloat('simulation_case', 'interest_rate'))

# Annualize investment costs
def init_pi_anu(model,kind):
    return (pi_inv_dic[kind]*model.interest_rate*((1+model.interest_rate)**model.L))/(((1+model.interest_rate)**model.L) - 1)

model.pi_anu = Param(model.RESnSSnCB,initialize=init_pi_anu)

def init_pi_inv(model,kind):
    return pi_inv_dic[kind]

model.pi_inv = Param(model.RESnSSnCB,initialize=init_pi_inv)

# Budget
model.pi_inv_bgt = Param(initialize=config_file.getint('budget', 'inv_bgt'))
model.pi_inv_bgt_L = Param(initialize=config_file.getint('budget', 'inv_bgt_L'))

# O&M costs of DG and CB
#€/kWh -> €/MWh
model.pi_om_PV = Param(initialize=eval(config_file.get('operation_maintenance_cost', 'om_cost_PV'))) 
model.pi_om_WD = Param(initialize=eval(config_file.get('operation_maintenance_cost', 'om_cost_WD')))
#€/kVAr -> €/MVArh
model.pi_om_CB = Param(initialize=eval(config_file.get('operation_maintenance_cost', 'om_cost_CB')))
# Not supplied energy cost
model.pi_ENS = Param(initialize=eval(config_file.get('operation_maintenance_cost', 'cost_ENS')))

    
# Prameters of CO2 emission
# Emission rate
nu_ss = config_file.getfloat('CO2', 'nu_SS')
nu_wd = config_file.getfloat('CO2', 'nu_WD')
nu_pv = config_file.getfloat('CO2', 'nu_PV')

nu_dg = {'SS': nu_ss, 'WD': nu_wd, 'PV': nu_pv}

def emission_rate(model, kind):
    return nu_dg[kind]

model.nu = Param(model.RESnSS, initialize=emission_rate)

# Price of emission
model.price_CO2 = Param(initialize=config_file.getfloat('CO2', 'price_CO2'))

# Increasing_emission_cost_factor
increasing_emission_cost_factor_ = config_file.getfloat('CO2', 'increasing_emission_cost_factor')
def increasing_emission_factor_init(model,t):
    return (1+increasing_emission_cost_factor_)**(t-1)
model.increasing_emission_factor = Param(model.Years,initialize=increasing_emission_factor_init)


# Subsidy rate
gamma_sup_wd = config_file.getfloat('Incentive', 'gamma_sup_WD')
gamma_sup_pv = config_file.getfloat('Incentive', 'gamma_sup_PV')

gamma_support = {'WD': gamma_sup_wd, 'PV': gamma_sup_pv}

def subsidy_rate(model, kind):
    return gamma_support[kind]

model.gamma_sup = Param(model.RES, initialize=subsidy_rate)

#pi_LOSS
#model.pi_LOSS = Param(initialize=68.60)

# Candidate buses to install DG 
canPV = system_data['candidate_PV'].tolist()
canWD = system_data['candidate_wd'].tolist()
canCB = system_data['candidate_CB'].tolist()

def canPV_(model,i):
    return canPV[i-1]

def canWD_(model,i):
    return canWD[i-1]

def canCB_(model,i):
    return canCB[i-1]

model.canPV = Param(model.LoadBuses,initialize=canPV_)
model.canWD = Param(model.LoadBuses,initialize=canWD_)
model.canCB = Param(model.LoadBuses,initialize=canCB_)


# Discount rate
model.discount_rate = Param(initialize=config_file.getfloat('simulation_case', 'discount_rate'))

# Increasing demand growth factor
increasing_load_factor_ = config_file.getfloat('simulation_case', 'increasing_load_factor')
def increasing_load_f_init(model,t):
    return (1+increasing_load_factor_)**(t-1)

model.increasing_load_f = Param(model.Years,initialize=increasing_load_f_init)

# Increasing energy cost factor
increasing_energy_cost_factor_ = config_file.getfloat('simulation_case', 'increasing_energy_cost_factor')
def increasing_energy_cost_f_init(model,t):
    return (1+increasing_energy_cost_factor_)**(t-1)

model.increasing_energy_cost_f = Param(model.Years,initialize=increasing_energy_cost_f_init)


# Power factor
if dg_dfig_vsi_mode:
    model.LEAD_LAG = Set(initialize=['lead','lag'])

    pf_ss = config_file.getfloat('DG_DFIG', 'pf_SS_DFIG')
    #lead
    pf_wd_lead = config_file.getfloat('DG_DFIG', 'pf_WD_lead_DFIG')
    pf_pv_lead = config_file.getfloat('DG_DFIG', 'pf_PV_lead_DFIG')

    #lag
    pf_wd_lag = config_file.getfloat('DG_DFIG', 'pf_WD_lag_DFIG')
    pf_pv_lag = config_file.getfloat('DG_DFIG', 'pf_PV_lag_DFIG')

    pf_dic_dg = {
              'WD':{'lead': pf_wd_lead, 'lag': pf_wd_lag},
              'PV':{'lead': pf_pv_lead, 'lag': pf_pv_lag},
              }

    def tan_init_lead(model, kind):
        return np.tan(np.arccos(pf_dic_dg[kind]['lead']))

    model.tan_phi_lead = Param(model.RES, initialize=tan_init_lead)

    def tan_init_lag(model, kind):
        return np.tan(np.arccos(pf_dic_dg[kind]['lag']))

    model.tan_phi_lag = Param(model.RES, initialize=tan_init_lag)

    pf_dic_ss = {'SS': pf_ss}

    def tan_init(model, kind):
        return np.tan(np.arccos(pf_dic_ss[kind]))

    model.tan_phi = Param(model.SS, initialize=tan_init)

else:
    pf_ss = config_file.getfloat('DG_DFIG', 'pf_SS')
    pf_wd = config_file.getfloat('DG_DFIG', 'pf_WD')
    pf_pv = config_file.getfloat('DG_DFIG', 'pf_PV')

    pf_dic = {'SS': pf_ss, 'WD': pf_wd, 'PV': pf_pv}

    def tan_init(model, kind):
        return np.tan(np.arccos(pf_dic[kind]))

    model.tan_phi = Param(model.RESnSS, initialize=tan_init)


# Load/Demand
def P_ld_n_init(model,bus):
    return system_data.loc[(system_data['Node']==bus).values,'P'].values[0] / (model.S_base.value)

model.P_ld_n = Param(model.LoadBuses,initialize=P_ld_n_init)

def Q_ld_n_init(model,bus):
    return system_data.loc[(system_data['Node']==bus).values,'Q'].values[0] / (model.S_base.value)

model.Q_ld_n = Param(model.LoadBuses,initialize=Q_ld_n_init)

# Resistance, Reactance, Impedance
def R_n_m_init(model,bus1,bus2):
    if bus1 >= bus2:
        bus = bus1
    else:
        bus = bus2

    a = model.Z_base.value

    return system_data.loc[(system_data['Node']==bus).values,'R'].values[0] / a

model.R_n_m = Param(model.Branches,initialize=R_n_m_init)

def X_n_m_init(model,bus1,bus2):
    if bus1 >= bus2:
        bus = bus1
    else:
        bus = bus2

    a = model.Z_base.value
    
    return system_data.loc[(system_data['Node']==bus).values,'X'].values[0] / a

model.X_n_m = Param(model.Branches,initialize=X_n_m_init)

def Z_n_m_init(model,bus1,bus2):
    # Already p.u.
    return np.sqrt(model.R_n_m[(bus1,bus2)]*model.R_n_m[(bus1,bus2)] + model.X_n_m[(bus1,bus2)]*model.X_n_m[(bus1,bus2)])

model.Z_n_m = Param(model.Branches, initialize=Z_n_m_init)


# Capacity of a DG unit  
model.P_PV_max = Param(initialize=eval(config_file.get('DG', 'P_PV_max'))/(model.S_base))
model.P_WD_max = Param(initialize=eval(config_file.get('DG', 'P_WD_max'))/(model.S_base))

if case == 'a':
    model.P_Node_max = Param(initialize=0.0) # No RES

else: #'b' or 'c'
    model.P_Node_max = Param(initialize=eval(config_file.get('DG', 'P_Node_max'))/(model.S_base)) # RES investment


# Capacity every substation expansion
model.S_SS_max = Param(initialize=eval(config_file.get('substation', 'S_SS_max'))/model.S_base )
# Maximum capacity of substation expansion
model.S_NEW_n_max = Param(initialize=eval(config_file.get('substation', 'S_NEW_n_max'))/model.S_base)

# Initial available substation capacity
SS_init = eval(config_file.get('substation', 'SS_init'))
def S_SS_n_init(model,bus):
    if bus == 1:
        return SS_init / model.S_base
    return 0.0

model.S_SS_n = Param(model.SSBuses,initialize=S_SS_n_init)

# Nominal Voltage and Voltage Limits
model.V_nom = Param(initialize=eval(config_file.get('voltage', 'V_nom')) / model.V_base)
model.V_max = Param(initialize=model.V_nom * config_file.getfloat('voltage', 'V_max'))
model.V_min = Param(initialize=model.V_nom * config_file.getfloat('voltage', 'V_min'))

# Transmission capacity
I_line_limit = eval(config_file.get('current', 'I_line_limit'))
def I_n_m_max_init(model,bus1,bus2):
    return I_line_limit / (model.S_base.value)

model.I_n_m_max = Param(model.Branches,initialize=I_n_m_max_init)

# Present worth factor
def pwf_init(model,t):
    return 1 / (1+model.discount_rate)**(t-1)

model.pwf = Param(model.Years, initialize=pwf_init)

### Mutable parameters
# Cost of purchased energy
model.pi_SS = Param(initialize=0.0, mutable=True)

# Initialize scenario-dependant variables
model.eta = Param(model.RESnld, initialize=0.0, mutable=True)

# Mutable parameter Time Block 
model.N_b_h = Param(initialize=0.0, mutable=True)

# Mutable parameter Scenario Probability 
model.proba = Param(initialize=0.0, mutable=True)

#======================================
#   Variables
#======================================

#FirstStageVariables

model.X_number_of = Var(model.Years,model.LoadBuses,model.RESnSSnCB,domain=NonNegativeIntegers)

model.InvestmentCostYear = Var(model.Years,domain=NonNegativeReals)

model.incentive = Var(model.Years,domain=NonNegativeReals)

#SecondStageVariables

model.pi_om = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_loss = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_ens = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_new = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_ss = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_cb = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_emi = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_emi_dg = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.pi_emi_ss = Var(model.Years,domain=NonNegativeReals, bounds=(0.0, None))

model.I_sqr = Var(model.Years,model.Branches,domain=NonNegativeReals, bounds=(0.0, None))

model.P_avl = Var(model.Years,model.LoadBuses,model.RES,domain=NonNegativeReals, bounds=(0.0, None))

model.Q_avl = Var(model.Years,model.LoadBuses,model.CB,domain=NonNegativeReals, bounds=(0.0, None))

model.P = Var(model.Years,model.LoadBuses,model.RESnSSnNS,domain=NonNegativeReals, bounds=(0.0, None))

model.P_to = Var(model.Years,model.Branches,domain=NonNegativeReals, bounds=(0.0, None))

model.P_in = Var(model.Years,model.Branches,domain=NonNegativeReals, bounds=(0.0, None))

if dg_dfig_vsi_mode:
    model.Q = Var(model.Years,model.LoadBuses,model.RESnSSnCB,domain=Reals)

else:
    model.Q = Var(model.Years,model.LoadBuses,model.RESnSSnCB,domain=NonNegativeReals, bounds=(0.0, None))

model.Q_to = Var(model.Years,model.Branches,domain=NonNegativeReals, bounds=(0.0, None))

model.Q_in = Var(model.Years,model.Branches,domain=NonNegativeReals, bounds=(0.0, None))

model.S_avl_SS = Var(model.Years,model.SSBuses,domain=NonNegativeReals, bounds=(0.0, None))

model.S_new = Var(model.Years,model.SSBuses,domain=NonNegativeReals, bounds=(0.0, None))

model.V_sqr = Var(model.Years,model.LoadBuses,domain=NonNegativeReals)

model.X_Pto = Var(model.Years,model.Branches,domain=Binary)

model.X_Pin = Var(model.Years,model.Branches,domain=Binary)

model.X_Qto = Var(model.Years,model.Branches,domain=Binary)

model.X_Qin = Var(model.Years,model.Branches,domain=Binary)

model.FirstStageCost = Var(domain=NonNegativeReals)

model.SecondStageCost = Var(domain=NonNegativeReals)


#======================================
#   Constraints
#======================================

# Installation Target
mu_pv = config_file.getfloat('DG_install_target', 'mu_PV')

def eq_target_pv(model,n):
    a = sum( [model.P_avl[years,n,'PV'] for n in model.LoadBuses])
    #b = sum( [model.increasing_load_f[years] * model.eta['ld'] * model.P_ld_n[n] for n in model.LoadBuses])
    b = sum( [model.increasing_load_f[years] * model.P_ld_n[n] for n in model.LoadBuses])
    return a >= mu_pv * b

#model.eq_target_pv = Constraint(model.LoadBuses,rule=eq_target_pv)


mu_wd = config_file.getfloat('DG_install_target', 'mu_WD')

def eq_target_wd(model,n):
    a = sum( [model.P_avl[years,n,'WD'] for n in model.LoadBuses])
    #b = sum( [model.increasing_load_f[years] * model.eta['ld'] * model.P_ld_n[n] for n in model.LoadBuses])
    b = sum( [model.increasing_load_f[years] * model.P_ld_n[n] for n in model.LoadBuses])
    return a >= mu_wd * b

#model.eq_target_wd = Constraint(model.LoadBuses,rule=eq_target_wd)


#P,Q
def eq1(model,t,n):
    if n == 1:
        return Constraint.Skip
    return model.P[t,n,'NS'] <= model.increasing_load_f[t] * model.eta['ld'] * model.P_ld_n[n]

model.eq1 = Constraint(model.Years,model.LoadBuses,rule=eq1)


def eq2(model,t,n):
    if n == 1:
        return Constraint.Skip
    return model.Q[t,n,'NS'] <= model.increasing_load_f[t] * model.eta['ld'] * model.Q_ld_n[n]

# Remove because capacitor bank constraints are used
#model.eq2 = Constraint(model.Years,model.LoadBuses,rule=eq2)

def eq3(model,t,m):
    e1 = sum([model.P_to[t,n,m] - model.P_in[t,n,m] for n,l in model.Branches if m ==l])
    e2 = sum([model.P_to[t,m,l] - model.P_in[t,m,l] + model.R_n_m[m,l]*model.I_sqr[t,m,l] for n,l in model.Branches if m == n])
    if m != 1:
        e3 = sum([model.P[t,m,RES] for RES in model.RESnNS])
        return e1 - e2 + e3 == model.increasing_load_f[t] * model.eta['ld'] * model.P_ld_n[m]
    else:
        e3 = model.P[t,m,'SS']
        return e1 - e2 + e3 == model.increasing_load_f[t] * model.eta['ld'] * model.P_ld_n[m]

model.eq3 = Constraint(model.Years,model.LoadBuses,rule=eq3)


def eq4(model,t,m):
    e1 = sum([model.Q_to[t,n,m] - model.Q_in[t,n,m] for n,l in model.Branches if m ==l])
    e2 = sum([model.Q_to[t,m,l] - model.Q_in[t,m,l] + model.X_n_m[m,l]*model.I_sqr[t,m,l] for n,l in model.Branches if m == n])
    if m != 1:#17
        e3 = sum([model.Q[t,m,RES] for RES in model.RESnCB])
        return e1 - e2 + e3 == model.increasing_load_f[t] * model.eta['ld'] * model.Q_ld_n[m]
    else:#18
        e3 = model.Q[t,m,'SS']
        return e1 - e2 + e3 == model.increasing_load_f[t] * model.eta['ld'] * model.Q_ld_n[m]

model.eq4 = Constraint(model.Years,model.LoadBuses,rule=eq4)


def V_sqr_bus1(model,t,n):
    if n == 1:
        return model.V_sqr[t,n].fix(substation_base_volt**2)
    else:
        return Constraint.Skip

#model.V_sqr_c = Constraint(model.Years,model.ScenarioIndexOfkthBlock,model.SSBuses,rule=V_sqr_bus1)

for i in model.V_sqr[:,1]:
    i.fix(substation_base_volt**2)


def eq5(model,t,m,n):
    e1 = model.R_n_m[m,n] * (model.P_to[t,m,n] - model.P_in[t,m,n]) + model.X_n_m[m,n] * (model.Q_to[t,m,n] - model.Q_in[t,m,n])
    return model.V_sqr[t,m] - 2 * e1 + model.Z_n_m[m,n] * model.Z_n_m[m,n]*model.I_sqr[t,m,n] - model.V_sqr[t,n] == 0.0

model.eq5 = Constraint(model.Years,model.Branches,rule=eq5)

def eq6_l(model,t,m):
    if m == 1:
        return  Constraint.Skip
    else:
        return model.V_min * model.V_min - model.V_sqr[t,m] <= 0.0 

model.eq6_l = Constraint(model.Years,model.LoadBuses,rule=eq6_l)

def eq6_r(model,t,m):
    if m == 1:
        return  Constraint.Skip
    else:
        return model.V_max * model.V_max - model.V_sqr[t,m] >= 0.0

model.eq6_r = Constraint(model.Years,model.LoadBuses,rule=eq6_r)

def eq7(model,t,m,n):
    return model.I_n_m_max[m,n] * model.I_n_m_max[m,n] - model.I_sqr[t,m,n] >= 0.0

model.eq7 = Constraint(model.Years,model.Branches,rule=eq7)

def eq8(model,t,m,n):
    return model.P_to[t,m,n] - model.V_nom * model.I_n_m_max[m,n] * model.X_Pto[t,m,n] <= 0.0

model.eq8 = Constraint(model.Years,model.Branches,rule=eq8)

def eq9(model,t,m,n):
    if reverse_case == 1:
        return model.P_in[t,m,n] <= 0.0
    elif reverse_case == 2:
        return model.P_in[t,m,n] - model.reverse_limit <= 0.0
    elif reverse_case == 3:
        return model.P_in[t,m,n] - model.V_nom * model.I_n_m_max[m,n] * model.X_Pin[t,m,n] <= 0.0
    else:
        raise Exception()

model.eq9 = Constraint(model.Years,model.Branches,rule=eq9)

def eq10(model,t,m,n):
    return model.Q_to[t,m,n] - model.V_nom * model.I_n_m_max[m,n] * model.X_Qto[t,m,n] <= 0.0

model.eq10 = Constraint(model.Years,model.Branches,rule=eq10)

def eq11(model,t,m,n):
    return model.Q_in[t,m,n] - model.V_nom * model.I_n_m_max[m,n] * model.X_Qin[t,m,n] <= 0.0

model.eq11 = Constraint(model.Years,model.Branches,rule=eq11)

def eq12(model,t,m,n):
    return model.X_Pto[t,m,n] + model.X_Pin[t,m,n] - 1 <= 0.0

model.eq12 = Constraint(model.Years,model.Branches,rule=eq12)

def eq13(model,t,m,n):
    return model.X_Qto[t,m,n] + model.X_Qin[t,m,n] - 1 <= 0.0

model.eq13 = Constraint(model.Years,model.Branches,rule=eq13)

def eq14(model,t,n):
    return model.P[t,n,'SS'] - model.S_avl_SS[t,n] / np.sqrt(1 + model.tan_phi['SS']**2) <= 0.0

model.eq14 = Constraint(model.Years,model.SSBuses,rule=eq14)

def eq15(model,t,n):
    return model.Q[t,n,'SS'] - model.tan_phi['SS'] * model.P[t,n,'SS'] <= 0.0

model.eq15 = Constraint(model.Years,model.SSBuses,rule=eq15)

def eq16(model,t,n):
    return model.S_new[t,n] - model.S_NEW_n_max <= 0.0

model.eq16 = Constraint(model.Years,model.SSBuses,rule=eq16)

def eq17(model,t,n):
    return model.S_avl_SS[t,n] - model.S_SS_n[n] - model.S_new[t,n] == 0.0

model.eq17 = Constraint(model.Years,model.SSBuses,rule=eq17)

def eq18(model,t,n):
    if t == 1:
        return model.S_new[t,n] - model.X_number_of[t,n,'SS'] * model.S_SS_max == 0.0
    else:
        return model.S_new[t,n] - model.X_number_of[t,n,'SS'] * model.S_SS_max - model.S_new[t-1,n] == 0.0

model.eq18 = Constraint(model.Years,model.SSBuses,rule=eq18)

def eq19(model,t,n):
    return model.P[t,n,'WD'] <= model.eta['WD'] * model.P_avl[t,n,'WD']

model.eq19 = Constraint(model.Years,model.LoadBuses,rule=eq19)

def eq20(model,t,n):
    return model.P[t,n,'PV'] <= model.eta['PV'] * model.P_avl[t,n,'PV']

model.eq20 = Constraint(model.Years,model.LoadBuses,rule=eq20)

def eq21(model,n):
    return sum([model.X_number_of[t_,n,'WD'] for t_ in model.Years]) - model.X_WD_n_max <= 0.0

model.eq21 = Constraint(model.LoadBuses,rule=eq21)

def eq22(model,n):
    return sum([model.X_number_of[t_,n,'PV'] for t_ in model.Years]) - model.X_PV_n_max <= 0.0

model.eq22 = Constraint(model.LoadBuses,rule=eq22)

def eq23(model,t,n):
    if t == 1:
        return model.P_avl[t,n,'WD'] - model.P_WD_max * model.X_number_of[t,n,'WD'] * model.canWD[n] == 0.0
    else:
        return model.P_avl[t,n,'WD'] - model.P_WD_max * model.X_number_of[t,n,'WD'] * model.canWD[n] -  model.P_avl[t-1,n,'WD'] == 0.0

model.eq23 = Constraint(model.Years,model.LoadBuses,rule=eq23)

def eq24(model,t,n):
    if t == 1:
        return model.P_avl[t,n,'PV'] - model.P_PV_max * model.X_number_of[t,n,'PV'] * model.canPV[n] == 0.0
    else:
        return model.P_avl[t,n,'PV'] - model.P_PV_max * model.X_number_of[t,n,'PV'] * model.canPV[n] -  model.P_avl[t-1,n,'PV'] == 0.0

model.eq24 = Constraint(model.Years,model.LoadBuses,rule=eq24)

def eq25(model,t,n):
    return model.Q[t,n,'CB'] <= model.Q_avl[t,n,'CB']

model.eq25 = Constraint(model.Years,model.LoadBuses,rule=eq25)

def eq26(model,n):
    return sum([model.X_number_of[t_,n,'CB'] for t_ in model.Years]) - model.X_CB_n_max <= 0.0

model.eq26 = Constraint(model.LoadBuses,rule=eq26)

def eq27(model,t,n):
    if t == 1:
        return model.Q_avl[t,n,'CB'] - model.Q_CB_max * model.X_number_of[t,n,'CB'] * model.canCB[n] == 0.0
    else:
        return model.Q_avl[t,n,'CB'] - model.Q_CB_max * model.X_number_of[t,n,'CB'] * model.canCB[n] -  model.Q_avl[t-1,n,'CB'] == 0.0

model.eq27 = Constraint(model.Years,model.LoadBuses,rule=eq27)



if dg_dfig_vsi_mode:
    def eq28_l(model,t,n):
        return model.P[t,n,'WD'] * model.tan_phi_lead['WD'] + model.Q[t,n,'WD'] >= 0.0

    model.eq28_l = Constraint(model.Years,model.LoadBuses,rule=eq28_l)

    def eq28_r(model,t,n):
        return model.Q[t,n,'WD'] - model.P[t,n,'WD'] * model.tan_phi_lag['WD'] <= 0.0

    model.eq28_r = Constraint(model.Years,model.LoadBuses,rule=eq28_r)

    def eq29_l(model,t,n):
        return model.P[t,n,'PV'] * model.tan_phi_lead['PV'] + model.Q[t,n,'PV'] >= 0.0

    model.eq29_l = Constraint(model.Years,model.LoadBuses,rule=eq29_l)

    def eq29_r(model,t,n):
        return model.Q[t,n,'PV'] - model.P[t,n,'PV'] * model.tan_phi_lag['PV'] <= 0.0

    model.eq29_r = Constraint(model.Years,model.LoadBuses,rule=eq29_r)

else:
    def eq28(model,t,n):
        return model.Q[t,n,'WD'] - model.P[t,n,'WD'] * model.tan_phi['WD'] <= 0.0

    model.eq28 = Constraint(model.Years,model.LoadBuses,rule=eq28)

    def eq29(model,t,n):
        return model.Q[t,n,'PV'] - model.P[t,n,'PV'] * model.tan_phi['PV'] <= 0.0

    model.eq29 = Constraint(model.Years,model.LoadBuses,rule=eq29)


def eq30(model,n):
    return model.P_Node_max - sum([model.P_WD_max * model.X_number_of[t,n,'WD'] + model.P_PV_max * model.X_number_of[t,n,'PV'] for t in model.Years]) >= 0.0

model.eq30 = Constraint(model.LoadBuses,rule=eq30)


def eq31(model,t):
    return model.InvestmentCostYear[t] - model.pi_inv_bgt <= 0.0

if case in ['a','b']:
    model.eq31 = Constraint(model.Years,rule=eq31)

def eq32(model):
    e1 = sum([model.pwf[t] * sum([model.pi_inv['SS'] * model.X_number_of[t,n,'SS'] for n in model.SSBuses]) for t in model.Years])
    e2 = sum([model.pwf[t] * sum([model.pi_inv['PV'] * model.X_number_of[t,n,'PV'] + model.pi_inv['WD'] * model.X_number_of[t,n,'WD'] + model.pi_inv['CB'] * model.X_number_of[t,n,'CB'] for n in model.LoadBuses if n != 1]) for t in model.Years])

    return e1 + e2 - model.pi_inv_bgt_L <= 0.0

if case in ['a','b']:
    model.eq32 = Constraint(rule=eq32)


# delta_P, delta_Q, slope_k, delta_S
# Piesewise linearization
model.number_H = Param(initialize=config_file.getint('piesewise_linearization', 'number_H'))
model.H = Set(initialize=range(1,(model.number_H.value +1)))

def delta_S_init(model,t,m,n,h):
    return (model.V_nom * model.I_n_m_max[m,n])/ float(model.number_H.value)

model.delta_S = Param(model.Years,model.Branches,model.H,initialize=delta_S_init)

def slope_k_init(model,t,m,n,h):
    return (2*h - 1) * model.delta_S[t,m,n,h]

model.slope_k = Param(model.Years,model.Branches,model.H,initialize=slope_k_init)

model.delta_P = Var(model.Years,model.Branches,model.H,within=NonNegativeReals, bounds=(0.0, None))
model.delta_Q = Var(model.Years,model.Branches,model.H,within=NonNegativeReals, bounds=(0.0, None))


def eq33(model,t,m,n):
    return model.V_nom*model.V_nom * model.I_sqr[t,m,n] - sum([model.slope_k[t,m,n,h]*model.delta_P[t,m,n,h] for h in model.H]) - sum([model.slope_k[t,m,n,h]*model.delta_Q[t,m,n,h] for h in model.H]) == 0.0

model.eq33 = Constraint(model.Years,model.Branches,rule=eq33)

def eq34(model,t,m,n):
    return model.P_to[t,m,n] + model.P_in[t,m,n] - sum([model.delta_P[t,m,n,h] for h in model.H]) == 0.0

model.eq34 = Constraint(model.Years,model.Branches,rule=eq34)

def eq35(model,t,m,n):
    return model.Q_to[t,m,n] + model.Q_in[t,m,n] - sum([model.delta_Q[t,m,n,h] for h in model.H]) == 0.0

model.eq35 = Constraint(model.Years,model.Branches,rule=eq35)

def eq36(model,t,m,n,h):
    return model.delta_P[t,m,n,h] - model.delta_S[t,m,n,h] <= 0.0

model.eq36 = Constraint(model.Years,model.Branches,model.H,rule=eq36)

def eq37(model,t,m,n,h):
    return model.delta_Q[t,m,n,h] - model.delta_S[t,m,n,h] <= 0.0

model.eq37 = Constraint(model.Years,model.Branches,model.H,rule=eq37)



#======================================
#   Equations for Objective
#======================================

def eq_investment_cost(model,t):
    if t == 1:
        return model.InvestmentCostYear[t] - sum([model.pi_anu['SS'] * model.X_number_of[t,n,'SS'] for n in model.SSBuses]) - sum([model.pi_anu['PV'] * model.X_number_of[t,n,'PV'] + model.pi_anu['WD'] * model.X_number_of[t,n,'WD'] + model.pi_anu['CB'] * model.X_number_of[t,n,'CB'] for n in model.LoadBuses if n != 1]) == 0.0
    else:
        return model.InvestmentCostYear[t] - sum([model.pi_anu['SS'] * model.X_number_of[t,n,'SS'] for n in model.SSBuses]) - sum([model.pi_anu['PV'] * model.X_number_of[t,n,'PV'] + model.pi_anu['WD'] * model.X_number_of[t,n,'WD'] + model.pi_anu['CB'] * model.X_number_of[t,n,'CB'] for n in model.LoadBuses if n != 1]) - model.InvestmentCostYear[t-1] == 0.0

model.eq_investment_cost = Constraint(model.Years,rule=eq_investment_cost)

def eq_incentive(model,t):
    return model.incentive[t] - sum([model.gamma_sup['WD'] * model.pi_inv['WD'] * model.X_number_of[t,n,'WD'] + model.gamma_sup['PV'] * model.pi_inv['PV'] * model.X_number_of[t,n,'PV']for n in model.LoadBuses if n != 1]) == 0.0

model.eq_incentive = Constraint(model.Years,rule=eq_incentive)

def eq_operation_maintenance_cost(model,t):
    return model.pi_om[t] - model.pi_loss[t] - model.pi_ens[t] - model.pi_ss[t] - model.pi_new[t] -model.pi_cb[t] - model.pi_emi[t] == 0.0

model.eq_operation_maintenance_cost = Constraint(model.Years,rule=eq_operation_maintenance_cost)

def eq_loss_cost(model,t):
    # Purchased cost is used instead of loss cost
    return model.pi_loss[t] - model.pi_SS * model.increasing_energy_cost_f[t] * sum([model.S_base/ (10.0**6) * model.R_n_m[n,m] * model.I_sqr[t,n,m] for n,m in model.Branches]) == 0.0

model.eq_loss_cost = Constraint(model.Years,rule=eq_loss_cost)


def eq_ens_cost(model,t):
    return model.pi_ens[t] - model.pi_ENS * sum([model.S_base / (10.0**6) * model.P[t,n,'NS'] for n in model.LoadBuses if n != 1]) == 0.0

model.eq_ens_cost = Constraint(model.Years,rule=eq_ens_cost)

def eq_purchased_cost(model,t):
    return model.pi_ss[t] - model.pi_SS * model.increasing_energy_cost_f[t] * sum([model.S_base / (10.0**6) * model.P[t,n,'SS'] for n in model.SSBuses]) == 0.0

model.eq_purchased_cost = Constraint(model.Years,rule=eq_purchased_cost)

def eq_dg_om_cost(model,t):
    return model.pi_new[t] - model.S_base / (10.0**6) * sum([model.pi_om_PV * model.P[t,n,'PV'] + model.pi_om_WD * model.P[t,n,'WD'] for n in model.LoadBuses if n != 1]) == 0.0

model.eq_dg_om_cost = Constraint(model.Years,rule=eq_dg_om_cost)

def eq_cb_om_cost(model,t):
    return model.pi_cb[t] - model.S_base / (10.0**6) * sum([model.pi_om_CB * model.Q[t,n,'CB'] for n in model.LoadBuses if n != 1]) == 0.0

model.eq_cb_om_cost = Constraint(model.Years,rule=eq_cb_om_cost)

def eq_emission_cost(model,t):
    return model.pi_emi[t] - model.pi_emi_dg[t] - model.pi_emi_ss[t] == 0.0

model.eq_emission_cost = Constraint(model.Years,rule=eq_emission_cost)

def eq_emission_cost_dg(model,t):
    return model.pi_emi_dg[t] - model.increasing_emission_factor[t] * model.S_base / (10.0**6) * model.price_CO2 * sum([model.nu['WD'] * model.P[t,n,'WD'] + model.nu['PV'] * model.P[t,n,'PV'] for n in model.LoadBuses if n != 1]) == 0.0

model.eq_emission_cost_dg = Constraint(model.Years,rule=eq_emission_cost_dg)

def eq_emission_cost_ss(model,t):
    return model.pi_emi_ss[t] - model.increasing_emission_factor[t] * model.S_base / (10.0**6) * model.price_CO2 * sum([model.nu['SS'] * model.P[t,n,'SS'] for n in model.SSBuses]) == 0.0

model.eq_emission_cost_ss = Constraint(model.Years,rule=eq_emission_cost_ss)

#======================================
#   Stage-specific cost computations
#======================================

def ComputeFirstStageCost_rule(model):
    return model.FirstStageCost - sum([model.pwf[t] * (model.InvestmentCostYear[t] - model.incentive[t]) for t in model.Years]) == 0.0

model.ComputeFirstStageCost = Constraint(rule=ComputeFirstStageCost_rule)


def ComputeSecondStageCost_rule(model):
    return model.SecondStageCost - sum([model.pwf[t] * model.proba * model.N_b_h * model.pi_om[t] for t in model.Years]) == 0.0

model.ComputeSecondStageCost = Constraint(rule=ComputeSecondStageCost_rule)

def total_cost_rule(model):
    # Causion: multiply len(scenario_list_all) by SecondStageCost because Pyomo automatically multipy scenario probability defined in ScenarioStructure.dat
    return (model.FirstStageCost + (len(scenario_list_all)*model.SecondStageCost))

model.TSC = Objective(rule=total_cost_rule, sense=minimize)



#======================================
# Assign stochastic data to mutable parameters
#======================================

def pysp_instance_creation_callback(scenario_name, node_names):
    #print 'Creating Scenario {} Instances'.format(scenario_name)

    #instance = model.create_instance()
    instance = model.clone()
    
    # Scenario data
    instance.eta.store_values(scenario_list_all[scenario_name]['factor'])
    
    # Cost of purchased energy
    instance.pi_SS.store_values(float(scenario_list_all[scenario_name]['cost_SS']))

    # The number of hours in the scenario
    instance.N_b_h.store_values(float(scenario_list_all[scenario_name]['hours']))

    # Scenario Probabilities
    instance.proba.store_values(float(scenario_list_all[scenario_name]['prob']))

    print 'Created {} instance'.format(scenario_name)

    return instance


