import pandas as pd
import numpy as np

def training_data(
    test_firstbatch, #your initial test data
    train_firstbatch, #your initial training data
    simulated_values, #results of all the simulations you have so far, including simulations that failed
    batch_id, #id of the next batch you want to simulate
    todays_date, #date of today, used for naming the batches
    path_batches, #path to all batches
):
    #assumes you results has the columns smiles and the metric you want to screen for
    #change this to your metric of interest
    #in this example you have the ionic conductivity in your property column and want to screen for the log(cond)
       
    smiles = []
    property = []
    smiles_failed = [] 
    for idx, row in simulated_values.iterrows():
        if row['property'] is not None:
            smiles.append(row['smiles'])
            property.append(np.log10(row['property']))
        else:
            smiles_failed.append(row['smiles'])
      
    assert len(smiles) == len(property)             
    results_simulated = pd.DataFrame(list(zip(smiles, property)), columns =['smiles', 'property'])
    
    data_prev = pd.read_csv(f'{test_firstbatch}') 
    train_prev = pd.read_csv(f'{train_firstbatch}')
    
    #merge your initial training data with all your previous simulations
    matching_rows = data_prev[data_prev['smiles'].isin(smiles)]
    results_simulated = pd.merge(results_simulated, matching_rows, on='smiles', how='left') 
    train_next = pd.concat([train_prev, results_simulated], ignore_index=True)
    train_next.to_csv(f'{path_batches}/batch{batch_id}_{todays_date}/train_batch{batch_id}.csv', index=False)
    
    #remove the molecules that you simulated or where simulations failed from your screening space
    data_next = data_prev[~data_prev['smiles'].isin(smiles)]
    data_cleaned = data_next[~data_next['smiles'].isin(smiles_failed)]
    data_cleaned.to_csv(f'{path_batches}/batch{batch_id}_{todays_date}/test_batch{batch_id}.csv', index=False)
    