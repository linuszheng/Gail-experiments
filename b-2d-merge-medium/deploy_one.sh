#!/bin/bash
echo "Training one GAIL instance"
echo "Screen name out-$1"
echo "Logging at experiments/out-$1.txt"
cp hyperparams.py experiments/out-$1-params.py
screen -Dm -L -Logfile experiments/out-$1.txt -S out-$1 python test_env.py -g $2
