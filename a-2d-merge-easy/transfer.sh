#!/bin/bash
echo "Retraining without overriding"
echo "Screen name out-$2(a)"
echo "Logging at experiments/out-$2.txt"
cp experiments/out-$1-params.py hyperparams.py
cp experiments/out-$1-params.py experiments/out-$2-params.py
cp /dev/null experiments/out-$2.txt
screen -dm -L -Logfile experiments/out-$2.txt -S "out-$2(a)" python test_env.py
