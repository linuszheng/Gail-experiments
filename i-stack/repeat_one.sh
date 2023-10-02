#!/bin/bash
echo "Retraining one GAIL instance"
echo "Screen name out-$1(d)"
echo "Logging at experiments/out-$1.txt"
cp experiments/out-$1-params.py hyperparams.py
rm experiments/out-$1.txt
screen -dm -L -Logfile experiments/out-$1.txt -S "out-$1(d)" python test_env.py
