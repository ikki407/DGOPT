#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Need to set simulation case." 1>&2
  echo "Please select from a, b, c." 1>&2
  exit 1
fi

# === Reading config file ===
CONFIG=config/general.conf
#gurobi_param=$(sed -n -e '/\[gurobi_parameters\]/,/*/p' ${CONFIG} |sed -e '1d;$d '| awk '$0 !~ /^#/{print $0}' | awk -F " *= *" '{print $1 "=" $2}')

gurobi_param=$(sed -n -e '/\[gurobi_parameters\]/,/*/p' ${CONFIG} | sed -e '1d;$d' | awk '$0 !~ /^#|\n/{print $0}')

gurobi_param_list=$(echo $gurobi_param | sed 's/\\n/\s/g')

# Simulation name
name="results"

# Case a
case_name=$1

echo "Simulation Case: ${case_name}"
echo "Results Directory: ${name}/"

runef -m concrete/ReferenceModel_${case_name}.py \
 --instance-directory scenario \
 --solver gurobi \
 --solution-writer=pyomo.pysp.plugins.csvsolutionwriter\
 --solver-options="${gurobi_param_list} " --solve --output-times \
 --compile-scenario-instances --output-solver-log \
 --output-file=efout_${case_name}.lp

# rename
mv ef.csv ${name}_${case_name}.csv
mv ef_StageCostDetail.csv ${name}_${case_name}_StageCostDetail.csv

# Arrange results
python arrange.py --case=${case_name} --results_folder=${name}
