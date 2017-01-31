# -*- coding: utf-8 -*-
"""
Create ScenarioStructure.dat for Two-Stage Stochastic Programming Problem of Pyomo 
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import argparse

parser = argparse.ArgumentParser(description='Create scenario file for pyomo optimization')
parser.add_argument('--bus_number', default=34, type=int, help='# of buses of used distribution system') 
args = parser.parse_args()


# Reading scenario data
with open('scenario_generation/scenario_list_all.pickle', 'rb') as handle:
    scenario_list_all = pickle.load(handle)


# Writing scenario information into ScenarioStructure.dat
with open('scenario/ScenarioStructure.dat', mode = 'w') as fh:
    fh.write('set Stages := \n')
    fh.write('\tFirstStage SecondStage;\n')
    fh.write('\n')

    # Node Name
    fh.write('set Nodes := \n')
    fh.write('\tRootNode\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenarioNode{}\n'.format(i))
    fh.write(';\n\n')
    
    # Node Stage
    fh.write('param NodeStage := \n')
    fh.write('\tRootNode FirstStage\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenarioNode{} SecondStage\n'.format(i))
    fh.write(';\n\n')

    # Node Parent & Child
    fh.write('set Children[RootNode] :=\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenarioNode{}\n'.format(i))
    fh.write(';\n\n')

    # Scenario probabilities
    scenario_prob = 1.0 / len(scenario_list_all.keys())
    fh.write('param ConditionalProbability := \n')
    fh.write('\tRootNode    1.0\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenarioNode{}    {}\n'.format(i, scenario_prob))
    fh.write(';\n\n')


    # Scenario Names
    fh.write('set Scenarios :=\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenario{}\n'.format(i))
    fh.write(';\n\n')


    # Allocate Scenario Names to each Scenario Leaf
    fh.write('param ScenarioLeafNode :=\n')
    fh.write('\n')

    for i in range(1, len(scenario_list_all.keys())+1):
        fh.write('\tScenario{} ScenarioNode{}\n'.format(i, i))
    fh.write(';\n\n')


    # Define First-stage variables
    fh.write('set StageVariables[FirstStage] :=\n')
    fh.write('\n')

    fh.write('\tInvestmentCostYear[*]\n')
    fh.write('\tincentive[*]\n')
    
    # Y_number_of
    fh.write('\tX_number_of[*,1,SS]\n')
    for i in ['WD', 'PV', 'CB']:
        for j in range(2, args.bus_number+1):
            fh.write('\tX_number_of[*,{},{}]\n'.format(j, i))
    fh.write(';\n\n')


    # Define Second-stage variables
    fh.write('set StageVariables[SecondStage] :=\n')
    fh.write('\n')

    fh.write('\tpi_om[*]\n')
    fh.write('\tpi_loss[*]\n')
    fh.write('\tpi_ens[*]\n')
    fh.write('\tpi_new[*]\n')
    fh.write('\tpi_ss[*]\n')
    fh.write('\tpi_cb[*]\n')
    fh.write('\tpi_emi[*]\n')
    fh.write('\tpi_emi_dg[*]\n')
    fh.write('\tpi_emi_ss[*]\n')
    fh.write('\tI_sqr[*,*,*]\n')

    # P    
    fh.write('\tP_avl[*,*,*]\n')

    fh.write('\tP[*,1,SS]\n')
    for i in ['WD', 'PV', 'NS']:
        for j in range(2, args.bus_number+1):
            fh.write('\tP[*,{},{}]\n'.format(j, i))

    fh.write('\tP_to[*,*,*]\n')
    fh.write('\tP_in[*,*,*]\n')

    # Q
    fh.write('\tQ_avl[*,*,*]\n')

    fh.write('\tQ[*,1,SS]\n')
    for i in ['WD', 'PV', 'CB']:
        for j in range(2, args.bus_number+1):
            fh.write('\tQ[*,{},{}]\n'.format(j, i))

    fh.write('\tQ_to[*,*,*]\n')
    fh.write('\tQ_in[*,*,*]\n')

    # Substation, voltage, and piesewise linearization
    fh.write('\tS_avl_SS[*,1]\n')
    fh.write('\tS_new[*,1]\n')
    fh.write('\tV_sqr[*,*]\n')
    fh.write('\tX_Pto[*,*,*]\n')
    fh.write('\tX_Pin[*,*,*]\n')
    fh.write('\tX_Qto[*,*,*]\n')
    fh.write('\tX_Qin[*,*,*]\n')
    fh.write('\tdelta_P[*,*,*,*]\n')
    fh.write('\tdelta_Q[*,*,*,*]\n')

    fh.write(';\n\n')

    # Set stage cost
    fh.write('param StageCost := \n')
    fh.write('\tFirstStage FirstStageCost\n')
    fh.write('\tSecondStage SecondStageCost\n')
    fh.write(';\n\n')

    # Set pyomo data format of stochastic programming, scenario-based or tree-based.
    fh.write('param ScenarioBasedData := True\n')
    fh.write(';\n\n')


