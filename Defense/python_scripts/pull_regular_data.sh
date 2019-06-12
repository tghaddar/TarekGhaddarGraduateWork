#!/bin/bash
linums="$(grep -n "Solve Time/Sweep" ubp_regular_suite.out | cut -d: -f1)"
echo $linums


