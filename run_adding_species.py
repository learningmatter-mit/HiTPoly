
import pandas as pd

from screening.screening_utils import find_latest_subdir

def run(
    path_batches='pathtoyourbatches',
    path_simulationdata='pathtoyoursimulationdata',
    newsmiles_path='pathwhereyousavenewsmilesotsimulate',
    molspercluster=6,
):
    latest_subdir = find_latest_subdir(path_batches)
    batch_id= int(latest_subdir.split('/')[-1].split('_')[0].replace('batch',''))
    
    #read in the files with all the suggested polymers for this batch
    mols_simulate = pd.read_csv(f'{path_batches}/{latest_subdir}/mols_simulate_{molspercluster}molspercluster.csv')[0].tolist()
    
    #read in the simulation data to check which simulations failed
    train_lastbatch = pd.read_csv(f"{latest_subdir}/train_batch{batch_id}.csv")
    simulation_data = pd.read_csv(path_simulationdata)
    smiles_prev = train_lastbatch['smiles']
    smiles_successful = []
    smiles_new_failed = []
    for idx, row in simulation_data.iterrows():
        if row['smiles'] not in smiles_prev and row['property'] is not None:
            smiles_successful.append(row['smiles'])
        else:
            smiles_new_failed.append(row['smiles']) 
            
    #smiles_new_failed contains all failed SMILES so we need to filter for the ones we simulated
    new_smiles_tosimulate = []
    for idx, row in mols_simulate.iterrows():
        if row['smiles'] in smiles_new_failed:
            new_smiles = mols_simulate.iloc[idx+1]['smiles']
            #check that you are not adding a SMILES that is from the next cluster and has already been simulated
            #since the first 6 SMILES belong to the first cluster, the next 6 SMILES belong to the second cluster and so on
            if new_smiles not in smiles_successful:
                new_smiles_tosimulate.append(new_smiles)

    new_smiles_tosimulate = pd.DataFrame(new_smiles_tosimulate, columns=['smiles'])
    new_smiles_tosimulate.to_csv(newsmiles_path, header=False, index=False)

if __name__=='__main__':
    run()