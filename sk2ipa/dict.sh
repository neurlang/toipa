#!/bin/bash

model=`ls -1 -t out/ | grep checkpoint | head -n 1 | tr -d ':'`
rm my2.model
ln -s -T "out/$model" my2.model

python3 code/run_dict.py --text=skfreq.txt  > skfreq.2023-11-03.txt
