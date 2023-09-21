#!/bin/bash
echo "Training one GAIL instance"
echo "Screen name out-$1(b)"
echo "Logging at experiments/out-$1.txt"
cp hyperparams.py experiments/out-$1-params.py
screen -dm -L -Logfile experiments/out-$1.txt -S "out-$1(b)" python test_env.py
