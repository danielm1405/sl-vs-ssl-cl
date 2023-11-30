#!/bin/bash

for SCRIPT in bash_files/longer-sequnces-of-tasks/*; do

    for SEED in 5 6 7;do
        SEED=$SEED python3 job_launcher.py --script $SCRIPT
    done

done
