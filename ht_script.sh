#!/bin/bash

export USER=$(whoami)

source 'pathtoyourenvironment'

export HITPOLY=/home/$USER/HiTPoly
export CLOUDPATH=/pathtoyourclustermounted

echo 'HT screening' `date`

succesful_simu=$(python $HITPOLY/return_successful_simus.py)
cd $CLOUDPATH/pathtoyoursimulationfolder
queue_simu="$(wc -l)"

combined_simu=$((queue_simu + succesful_simu))
cutoff=14 #you can adjust to the minimum number of successful simulations before the next batch

echo 'This batch has ' $succesful_simu ' successful simulations and ' $queue_simu ' in the queue'

if [ "$queue_simu" -eq "0" -a "$succesful_simu" -gt "$cutoff" ] ; then
    echo 'Retraining GPR on the ' `date`
    echo 'after ' $succesful_simu
    python $HITPOLY/run_screening.py --path_batches path/to/your/batch_data
elif [ "$combined_simu" -lt "$cutoff" ] ; then
    echo 'Adding Species on the ' `date`
    python $HITPOLY/run_adding_species.py
fi

#take the species selected by the GPR and run the simulations
echo 'Running simulations on the ' `date`
#you should add a job submission command here
#don't forgot to also submit when you add new species if the first polymer of the cluster failed
#read in the smiles you screened and pass them to run_box_builder.py
#then execute run_analysis.py 
