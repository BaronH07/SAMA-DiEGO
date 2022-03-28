#!/bin/sh
# Delete comments if needed
alg=9 #Run with SAMA-DiEGO-A (PV + MI-ES), if run with B, C, D please set to 10, 11, 12, accordingly
func=2 # Function id, the second choice of 'Func_ids' in IOHPBO_config.json
dim=1 # The first choice of 'Dims' in IOHPBO_config.json, the dimensionality
# run: stands for count of repetition
# Example exclusively experiment on the eighth problem in IOHPBO_config.json with 11 independent runs
for run in 1 3 5 7 9 11
do
  if [ $run -eq 11 ]
    then
      python run_IOHPBO_test.py -F $func -A $alg -R $run -D $dim
    else
      next_r=$(( run + 1 ))
      python run_IOHPBO_test.py -F $func -A $alg -R $run -D $dim &
      python run_IOHPBO_test.py -F $func -A $alg -R $next_r -D $dim
    fi
done

# Run multiple problems together by specifying the function id here
for func in 8 9 10 11 12
do
  for run in 1 3 5 7 9 11 # Number of runs
  do
    if [ $run -eq 11 ]
      then
        python run_IOHPBO_test.py -F $func -A $alg -R $run -D $dim
      else
        next_r=$(( run + 1 ))
        python run_IOHPBO_test.py -F $func -A $alg -R $run -D $dim &
        python run_IOHPBO_test.py -F $func -A $alg -R $next_r -D $dim
      fi
  done
done
