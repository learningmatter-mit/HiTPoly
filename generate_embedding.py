import pandas as pd
import os
import torch
from datetime import datetime
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from screening.embedding import generate_longsmiles


def generate_embedding(smiles):
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    
    inputs = tokenizer(smiles, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        
    save_output = outputs.pooler_output
    
    return save_output

def run(
    smiles,
    test_data_path, #should be pandas dataframe with column 'smiles'
    train_data_path, #should be pandas dataframe with column 'smiles' and the screening metric you want to use 'property'
    path_batches,
):
    
    todays_date = datetime.today().strftime('%y%m%d')

    if not os.path.exists(f"{path_batches}/batch0"):
        os.makedirs(f"{path_batches}/batch0")

    test_data = pd.read_csv(test_data_path)
    smiles_test, longsmiles_test = generate_longsmiles(test_data['smiles'].values)
    
    train_data = pd.read_csv(train_data_path)
    smiles_train, longsmiles_train = generate_longsmiles(train_data['smiles'].values)
    
    embeddings_test = generate_embedding(longsmiles_test)
    embeddings_train = generate_embedding(longsmiles_train)

    #generate PCA features for test data and apply this embedding to the training data
    pca = PCA(n_components=50, random_state=42)
    transformed_test = pca.fit_transform(embeddings_test)
    
    var = pca.explained_variance_ratio_
    print(f"Explained variance of PCA on test data is {sum(var)}")

    df_test = pd.DataFrame(transformed_test.cpu().numpy(), columns=[f"feat{i}" for i in range(0, 50)])
    df_test['long_smiles'] = longsmiles_test
    df_test['smiles'] = smiles_test
    df_test.to_csv(f"{path_batches}/batch0/test_batch0.csv", index=False)

    transformed_train = pca.transform(embeddings_train)
    df_train = pd.DataFrame(transformed_train.cpu().numpy(), columns=[f"feat{i}" for i in range(0, 50)])
    df_train['long_smiles'] = longsmiles_train
    df_train['smiles'] = smiles_train
    df_train['property'] = train_data['property']  
    df_train.to_csv(f"{path_batches}/batch0/train_batch0.csv", index=False)