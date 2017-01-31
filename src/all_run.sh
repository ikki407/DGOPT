#!/bin/bash


# === Reading config file ===
CONFIG=config/general.conf

# === Scenario Generation ===
method=$(awk -F ' *= *' '$1=="method"{print $2}' "${CONFIG}")
data=$(awk -F ' *= *' '$1=="data"{print $2}' "${CONFIG}")

if [ $method = "kmeans" ]; then
    step1=$(awk -F ' *= *' '$1=="step1"{print $2}' "${CONFIG}")
    step2=$(awk -F ' *= *' '$1=="step2"{print $2}' "${CONFIG}")
    step3=$(awk -F ' *= *' '$1=="step3"{print $2}' "${CONFIG}")
    python scenario_generation/scenario_generation.py --step1=$step1 --step2=$step2 --step3=$step3 --data=$data

elif [ $method = "kde" ]; then
    scenario_num=$(awk -F ' *= *' '$1=="scenario_num"{print $2}' "${CONFIG}")
    watch_years=$(awk -F ' *= *' '$1=="watch_years"{print $2}' "${CONFIG}")
    python scenario_generation/scenario_generation_kde.py --scenario_num=$scenario_num --data=$data --watch_years=$watch_years

elif [ $method = "duration" ]; then
    step1=$(awk -F ' *= *' '$1=="step1"{print $2}' "${CONFIG}")
    step2=$(awk -F ' *= *' '$1=="step2"{print $2}' "${CONFIG}")
    step3=$(awk -F ' *= *' '$1=="step3"{print $2}' "${CONFIG}")
    python scenario_generation/scenario_generation_duration.py --step1=$step1 --step2=$step2 --step3=$step3 --data=$data

else
    echo 'Set scenario generation method from (kmeans, kde, duration) in general.conf' 1>&2
    exit 1  
fi


# === Make Scenario data file ===
bus_number=$(awk -F ' *= *' '$1=="bus_number"{print $2}' "${CONFIG}")
python scenario/make_scenariofile.py --bus_number=$bus_number



# === Optimization ===
case=$(awk -F ' *= *' '$1=="case"{print $2}' "${CONFIG}")

if [ $case = "all" ]; then
    sh run.sh a
    sh run.sh b
    sh run.sh c
elif [ $case = "a" ]; then
    sh run.sh a
elif [ $case = "b" ]; then
    sh run.sh b
elif [ $case = "c" ]; then
    sh run.sh c
else
    echo 'Set correct simulation case from (all, a, b, c) in general.conf' 1>&2
    exit 1  
fi

# === Visualization of results ===
case=$(awk -F ' *= *' '$1=="case"{print $2}' "${CONFIG}")

if [ $case = "all" ]; then
    python visualize_siting_sizing.py --bus_number=$bus_number
fi

# === Postprocessing for easy-to-see ===
python postprocessing.py
