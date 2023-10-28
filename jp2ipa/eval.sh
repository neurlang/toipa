#!/bin/bash

model=`ls -1 -t out/ | grep checkpoint | head -n 1 | tr -d ':'`
rm my2.model
ln -s -T "out/$model" my2.model

awk '{print tolower($0)}' < 1.txt > 1l.txt

python3 code/run_evaluate.py --text=1l.txt --orig=no --old=yes | sed 's/ˈ//g'
#python3 code/run_evaluate.py --text=1l.txt --orig=yes --old=no
#python3 code/run_evaluate.py --text=2.txt --orig=no --old=yes

