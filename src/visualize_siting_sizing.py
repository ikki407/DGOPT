# -*- coding: utf-8 -*-
"""
Visualizing all results in one image.

Cumulative installed capacity of substaion, wind turbine, PV, and capacitor bank.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
#import seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import cm
import numpy as np
import argparse
import ConfigParser
import cv2

pd.options.mode.chained_assignment = None

print '##################################################\n'
print '         Visualization of all results             \n'
print '##################################################\n'


# === ArgParser ===
parser = argparse.ArgumentParser(description='Create visualization of all results')
parser.add_argument('--bus_number', default=34, type=int, help='Number of buses of used distribution system') 
args = parser.parse_args()

# === ConfigParser
config_file = ConfigParser.SafeConfigParser()
config_file.read('config/parameter.conf')

# === Distribution System of Simulation ===
distribution_system_bus = args.bus_number

# === Matplotlib setting ===
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

# === Substation and DG unit capacity ===
P_WD_max = eval(config_file.get('DG', 'P_WD_max')) / 10.0**3
P_PV_max = eval(config_file.get('DG', 'P_PV_max')) / 10.0**3
Q_CB_max = eval(config_file.get('DG', 'Q_CB_max')) / 10.0**3
S_SS_max = eval(config_file.get('substation', 'S_SS_max')) / 10.0**3



cases = ['a', 'b', 'c']

col_names = ['Stage','Node','Var','index','value']
result_a = pd.read_csv('results_a.csv', names=col_names)
result_b = pd.read_csv('results_b.csv', names=col_names)
result_c = pd.read_csv('results_c.csv', names=col_names)

data_dict = {'a': result_a, 'b': result_b, 'c':result_c}


fig = plt.figure(figsize=(45,36))
bus_list = []
for idx, each_case in enumerate(cases):
    ax1 = plt.subplot2grid((3,4), (idx,0))
    ax2 = plt.subplot2grid((3,4), (idx,1))
    ax3 = plt.subplot2grid((3,4), (idx,2))
    ax4 = plt.subplot2grid((3,4), (idx,3))
    # reverse power flow
    a = data_dict[each_case]

    # Cumulative installed capacity of substation, WD, PV, and CB.
    res_power = a.iloc[(a.Var==' X_number_of').values,:]
    # 
    res_power['year'] = res_power['index'].apply(lambda x: int(x.split(':')[0]))
    res_power['bus'] = res_power['index'].apply(lambda x: int(x.split(':')[1]))
    res_power['res'] = res_power['index'].apply(lambda x: x.split(':')[2])

    # SS
    res_power_ss = res_power.iloc[(res_power['res']=="'SS'").values, :]
    res_power_ss = res_power_ss.groupby(['year', 'bus'])['value'].sum().unstack()
    res_power_ss = res_power_ss.apply(np.round)
    res_power_ss = res_power_ss.iloc[:,(res_power_ss!=0).any().values]
    res_power_ss *= S_SS_max

    # WD
    res_power_wd = res_power.iloc[(res_power['res']=="'WD'").values, :]
    res_power_wd = res_power_wd.groupby(['year', 'bus'])['value'].sum().unstack()
    res_power_wd = res_power_wd.apply(np.round)
    res_power_wd = res_power_wd.iloc[:,(res_power_wd!=0).any().values]
    res_power_wd *= P_WD_max

    # PV
    res_power_pv = res_power.iloc[(res_power['res']=="'PV'").values, :]
    res_power_pv = res_power_pv.groupby(['year', 'bus'])['value'].sum().unstack()
    res_power_pv = res_power_pv.apply(np.round)
    res_power_pv = res_power_pv.iloc[:,(res_power_pv!=0).any().values]
    res_power_pv *= P_PV_max

    # CB
    res_power_cb = res_power.iloc[(res_power['res']=="'CB'").values, :]
    res_power_cb = res_power_cb.groupby(['year', 'bus'])['value'].sum().unstack()
    res_power_cb = res_power_cb.apply(np.round)
    res_power_cb = res_power_cb.iloc[:,(res_power_cb!=0).any().values]
    res_power_cb *= Q_CB_max

    # Visualization
    bus_shape = max(res_power_ss.shape[1], res_power_wd.shape[1], res_power_pv.shape[1], res_power_cb.shape[1])
    bus_shape = float(bus_shape)

    # SS
    tmp = res_power_ss.cumsum()
    bottom = 0
    for i in tmp.columns:
        bus_list.append(i)
        ax1.bar(tmp.index.values, tmp.loc[:,i], bottom=bottom, color=cm.jet(1.*i/distribution_system_bus), width=0.5, label='bus{}'.format(i))
        bottom += tmp.loc[:,i]

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_xlabel('years', fontsize=24)
    ax1.set_ylabel('Capacity (kW)', fontsize=24)
    ax1.set_title('Cumulative expansion of substation at each year')


    # WD
    tmp = res_power_wd.cumsum()
    bottom = 0
    for i in tmp.columns:
        bus_list.append(i)
        ax2.bar(tmp.index.values, tmp.loc[:,i], bottom=bottom, color=cm.jet(1.*i/distribution_system_bus), label='bus{}'.format(i))
        bottom += tmp.loc[:,i]

    handles, labels = ax2.get_legend_handles_labels()
    ax2.set_xlabel('years', fontsize=24)
    ax2.set_ylabel('Capacity (kW)', fontsize=24)
    ax2.set_title('Cumulative installed capacity of wind power at each year')


    # PV
    tmp = res_power_pv.cumsum()
    bottom = 0
    for i in tmp.columns:
        bus_list.append(i)
        ax3.bar(tmp.index.values, tmp.loc[:,i], bottom=bottom, color=cm.jet(1.*i/distribution_system_bus), label='bus{}'.format(i))
        bottom += tmp.loc[:,i]

    handles, labels = ax3.get_legend_handles_labels()
    ax3.set_xlabel('years', fontsize=24)
    ax3.set_ylabel('Capacity (kW)', fontsize=24)
    ax3.set_title('Cumulative installed capacity of PV at each year')


    # CB
    tmp = res_power_cb.cumsum()
    bottom = 0
    for i in tmp.columns:
        bus_list.append(i)
        ax4.bar(tmp.index.values, tmp.loc[:,i], bottom=bottom, color=cm.jet(1.*i/distribution_system_bus), label='bus{}'.format(i))
        bottom += tmp.loc[:,i]

    handles, labels = ax4.get_legend_handles_labels()
    ax4.set_xlabel('years', fontsize=24)
    ax4.set_ylabel('Capacity (kVar)', fontsize=24)
    ax4.set_title('Cumulative installed capacity of CB at each year')

# Arrange xlim and y_lim
ax_list = fig.get_axes()

# Arrange all y_lim
ylim_max = 0
for i in ax_list:
    ylim_max = max(ylim_max, i.get_ybound()[1])
for i in ax_list:
    i.set_ylim(0, ylim_max)

# Arrange all xlim
ax_list = fig.get_axes()
max_xlim = config_file.getint('simulation_case', 'years') + 2
# get
for ax_list_ in ax_list:
    ax_list_.set_xlim(1, max_xlim)

case_list = ['A', 'B', 'C']
for i in xrange(len(ax_list)):
    text_list = ['Case {}: SUB'.format(case_list[i//4]), 'Case {}: WIND'.format(case_list[i//4]), 'Case {}: PV'.format(case_list[i//4]), 'Case {}: CB'.format(case_list[i//4])]
    ax_list[i].text(1, ylim_max-100, text_list[i%4], ha = 'left', va = 'top', fontsize=50, fontweight='normal') 
    ax_list[i].tick_params(labelsize=24)

fig.tight_layout()
plt.savefig('visualize/all_results_siting_sizing.png')

# Make legend by using bus_list
bus_list = list(set(bus_list))
bus_list.sort()

fig = plt.figure(figsize=(5,20+(args.bus_number - 34)/10.0))
axes = fig.add_subplot(1, 1, 1, axisbg='white')
yidx = 1
for i in bus_list:
    axes.barh(yidx, 0.5, color=cm.jet(1.*i/distribution_system_bus))
    axes.text(1.0, yidx+0.2, 'bus{}'.format(i), horizontalalignment='center', fontsize=42)
    axes.set_xlim(0.0, 2.0)
    yidx += 1
    axes.set_xticklabels([])
    axes.set_yticklabels([])
plt.savefig('visualize/legend_siting_sizing.png')

# Concat two images
img1 = cv2.imread('visualize/all_results_siting_sizing.png')
img2 = cv2.imread('visualize/legend_siting_sizing.png')

if img1.shape[0] < img2.shape[0]:
    img_diff = img2.shape[0]-img1.shape[0]
    img2 = img2[img_diff:,:,:]
    img3 = cv2.hconcat([img1, img2])
    cv2.imwrite('visualize/visualize_results_siting_sizing.png', img3)
    
else:
    white_img = np.zeros((img1.shape[0]-img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
    white_img.fill(255)
    img2 = cv2.vconcat([img2, white_img])    
    img3 = cv2.hconcat([img1, img2])
    cv2.imwrite('visualize/visualize_results_siting_sizing.png', img3)

print 'Done'
