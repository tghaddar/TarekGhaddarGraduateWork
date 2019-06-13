#!/bin/bash
linums="$(grep -n "Solve Time/Sweep" ubp_regular_suite_fcfs.out | cut -d: -f1)"
solvepersweep=()
solveperunknown=()
ctr=0
for l in $linums
do
  newline=$((l+2))
  curline=$(sed -n ${newline}p ubp_regular_suite_fcfs.out)
  IFS=' ' read -r -a array <<< $curline
  item=${array[5]}
  item2=${array[6]}
  solvepersweep+=($item)
  solveperunknown+=($item2)
done

printf "%s\n" ${solvepersweep[@]} > solvepersweep_regular_fcfs.txt
printf "%s\n" ${solveperunknown[@]} > solveperunknown_regular_fcfs.txt
