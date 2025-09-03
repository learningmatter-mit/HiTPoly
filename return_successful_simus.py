"""
this script returns the amount of successful simulations per batch
WARNING!!!! find_latest_subdir respons the last time a dir was EDITED so make sure to not edit the older batches
""" 

import pandas as pd

from screening.screening_utils import find_latest_subdir

def run(
    path_batches='pathtoyourbatches',
    path_simulationdata='pathtoyoursimulationdata'
):
    #search all dirs in the batch folder and find newest folder
    latest_subdir = find_latest_subdir(path_batches)
    batch_id= int(latest_subdir.split('/')[-1].split('_')[0].replace('batch',''))

    #check how many new succesful simus 
    train_lastbatch = pd.read_csv(f"{latest_subdir}/train_batch{batch_id}.csv")
    
    simulation_data = pd.read_csv(path_simulationdata)
    
    smiles_prev = train_lastbatch['smiles']
    
    smiles_new = []
    for idx, row in simulation_data.iterrows():
        if row['smiles'] not in smiles_prev and row['property'] is not None:
            smiles_new.append(row['smiles'])

    succesful_simu = len(smiles_new) - len(train_lastbatch)
    return succesful_simu
    
if __name__=='__main__':
    succesful_simu = run()
    print(succesful_simu)