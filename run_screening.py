import argparse
from datetime import datetime
import pandas as pd
import os
from screening.screening_utils import find_latest_subdir, preprocess_data, get_fitted_model, clustering_data, find_mols_tosimulate
from screening.metrics import training_data


import random
random.seed(42)

def run(
    path_batches,
    train_firstbatch_path,
    path_simulationdata,
    elbow=False,
    cluster_data=True,
    clusters=20,
    molspercluster=6, #this tag controls how many molecules per cluster are added to be potentially simulated
):
    todays_date =  datetime.today().strftime('%y%m%d')
    latest_subdir = find_latest_subdir(path_batches)
    prev_batch= int(latest_subdir.split('/')[-1].split('_')[0].replace('batch','')) 
    batch_id = prev_batch+1
    
    
    directory=f'{path_batches}/batch{batch_id}_{todays_date}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    test_firstbatch_path = f"{path_batches}/batch0/test_batch0.csv"
    
    train_firstbatch = pd.read_csv(train_firstbatch_path)
    simulated_values = pd.read_csv(path_simulationdata)

    training_data(
        test_firstbatch=test_firstbatch_path,
        train_firstbatch=train_firstbatch,
        simulated_values=simulated_values,
        batch_id=batch_id,
        todays_date=todays_date,
        path_batches=path_batches,
        )
  

    train_data_path = f'{path_batches}/batch{batch_id}_{todays_date}/train_batch{batch_id}.csv'
    test_data_path = f'{path_batches}/batch{batch_id}_{todays_date}/test_batch{batch_id}.csv'
    
    features_scaled, property, features_scaled_eval = preprocess_data(train_data_path=train_data_path, 
                                                                      test_data_path=test_data_path,
                                                                      )

    model = get_fitted_model(features_scaled, property.view(-1,1))

    if cluster_data:
        cluster_labelling = clustering_data(
            test_data=features_scaled_eval, 
            clusters=clusters, 
            elbow = elbow)
        
        new_mols = find_mols_tosimulate(
            amount=molspercluster, 
            test_data_path=test_data_path,
            features_scaled_eval=features_scaled_eval,
            cluster_labelling=cluster_labelling,
            model=model,
            property=property,
            clusters=clusters,
        )
        
    new_mols.to_csv(f'{path_batches}/batch{batch_id}_{todays_date}/mols_simulate_{molspercluster}molspercluster.csv', header=False, index=False)
    
    print("Saved new smiles to simulate")
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run GPR for high throughput screening")
    parser.add_argument(
        "-p", "--path_batches", help="Path to the where the batches are saved"
    )
    parser.add_argument(
        "-fb", "--train_firstbatch_path", help="Path to the initial training data"
    )
    parser.add_argument(
        "-sd", "--path_simulationdata", help="Path to all the simulation data including smiles of failed simulations"
    )

    args = parser.parse_args()
    
    run(
        path_batches=args.path_batches,
        train_firstbatch_path=args.train_firstbatch_path,
        path_simulationdata=args.path_simulationdata
    )