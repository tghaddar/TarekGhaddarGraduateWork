#!/bin/bash
linums="$(grep -n "Solve Time/Sweep" ubp_lb_suite_fcfs.out | cut -d: -f1)"
solvepersweep=()
solveperunknown=()
ctr=0
for l in $linums
do
  newline=$((l+2))
  curline=$(sed -n ${newline}p ubp_lb_suite_fcfs.out)
  IFS=' ' read -r -a array <<< $curline
  item=${array[5]}
  item2=${array[6]}
  solvepersweep+=($item)
  solveperunknown+=($item2)
done
echo ${solvepersweep[@]}
echo ${solveperunknown[@]}

printf "%s " ${solvepersweep[@]} > solvepersweep_lb_fcfs.txt
printf "%s " ${solveperunknown[@]} > solveperunknown_lb_fcfs.txt
