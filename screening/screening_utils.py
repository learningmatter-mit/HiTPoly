import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition.analytic import _scaled_improvement, _log_ei_helper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def find_latest_subdir(path_batches):
    all_subdirs = []
    for d in os.listdir(f"{path_batches}"):
        bd = os.path.join(f"{path_batches}", d)
        if os.path.isdir(bd): all_subdirs.append(bd)
        
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir


def read_line_by_index(file_path, index):
    skip_rows = list(range(1, index)) + list(range(index + 1, float('inf')))
    df = pd.read_csv(f"{file_path}", skiprows=skip_rows)
    return df.iloc[0]

def preprocess_data(
    train_data_path,
    test_data_path,
):
    data_train = pd.read_csv(f"{train_data_path}")
    data_eval = pd.read_csv(f"{test_data_path}")
    
    property = data_train['property'].values
    property = torch.from_numpy(property)
    
    features = data_train.drop(columns=['smiles', 'long_smiles', 'property']).values
    scaler = StandardScaler().fit(features)
    features_scaled = scaler.transform(features)
    features_scaled = torch.from_numpy(features_scaled)
    
    features_eval = data_eval.drop(columns=['smiles', 'long_smiles']).values
    features_scaled_eval = scaler.transform(features_eval)
    features_scaled_eval = torch.from_numpy(features_scaled_eval)
    
    return features_scaled, property, features_scaled_eval

def clusterkmeans(
    data,
    clusters,
    elbow,
):
    #if set to true plots the elbow heuristic
    if elbow:
        from yellowbrick.cluster import KElbowVisualizer
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, 50))

        visualizer.fit(data) 
        visualizer.show()
        plt.savefig('./gpr/data/KElbowVisualizer.png')
        
    print(f"Clustering kmeans++ manhattan distance for {clusters} clusters")        

    list_data = []
    for i in range(data.shape[0]):
        list_data.append(list(data[i]))
    #breakpoint()
    
    initial_centers = kmeans_plusplus_initializer(list_data, clusters).initialize()
    kmeans_instance = kmeans(list_data, initial_centers, metric=distance_metric(type_metric.MANHATTAN))
    
    kmeans_instance.process()
    cluster_labelling = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()

    return cluster_labelling, final_centers

def clustering_data(
    test_data,
    clusters,
    elbow,
):
    cluster_labelling, final_centers = clusterkmeans(data=test_data,clusters=clusters, elbow=elbow)
    print("Retrieved cluster assignments")        
            
    return cluster_labelling

def get_fitted_model(train_x, train_obj, kernel="rbf",state_dict=None):
    # initialize and fit model
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": train_x.shape[-1]} # Some hyperparameters for the Mateern kernel
    if kernel == "matern":
        base_kernel = MaternKernel(**kernel_kwargs) # Classic matern kernel
    elif kernel == "rff":
        base_kernel = RFFKernel(**kernel_kwargs, num_samples=1024) # Random Fourier Features with the RBFKernel
    elif kernel == "rbf":
        rbf = ScaleKernel(RBFKernel())
        rbf.outputscale = 0.5
        rbf.base_kernel.lengthscale = 1
        base_kernel = rbf 
    else:
        ValueError(f"Kernel not defined correctly: {kernel}")
    covar_module = ScaleKernel(base_kernel) 
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-2)) # Theoretical hyperparameters for the noise

    model = SingleTaskGP(
        train_X=train_x, 
        train_Y=train_obj,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
        covar_module=covar_module,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_mll(mll)
    return model  

def _mean_and_sigma(model, X_test):
    min_var: float = 1e-12
    posterior = model.posterior(
        X=X_test, posterior_transform=None
    )
    mean = posterior.mean.squeeze(-2).squeeze(-1) 
    sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
    return mean, sigma

def expected_improvement(model, X_test, best_f, maximize=True):
    """
    Output of this is the value for the expecation of improvement
    The largest value of improvement will be the argmax for the best predicted sample
    """
    mean, sigma = _mean_and_sigma(model, X_test)
    u = _scaled_improvement(mean, sigma, best_f, maximize)
    return _log_ei_helper(u) + sigma.log()    

def reasonable_smiles(smiles):
    if "Br" in smiles:
        return False
    elif "C1CO1" in smiles or "C2CO2" in smiles or "C3CO3" in smiles or "C4CO4" in smiles or "C5CO5" in smiles:
        return False
    else:
        return True

#finds in each cluster the 2 points with highes expected improvement that fullfill certain criteria
def find_mols_tosimulate(
    test_data_path,
    features_scaled_eval,
    cluster_labelling,
    model,
    property,
    clusters,
    amount,
    ):
    smiles_sim = []
    long_smiles_sim = []
    
    test_data_df = pd.read_csv(f"{test_data_path}")    
    test_data_df['id'] = test_data_df.index
    
    exped_improvement = expected_improvement(model, features_scaled_eval, property.min(), maximize=True)
    exped_improvement = exped_improvement.detach().numpy()
    test_data_df['EI'] = exped_improvement

    for i in range(clusters):
        ids = cluster_labelling[i]
        df_cluster = test_data_df[test_data_df['id'].isin(ids)]
        df_cluster = df_cluster.sort_values(by=['EI'], ascending=False).reset_index()

        count_smiles = 0
        count_df = 0
        while count_smiles < amount:
            smils = df_cluster.loc[count_df, 'smiles']
            #this is an optional flag to filter out molecules that are not appropiate for your application
            if reasonable_smiles(smils): 
                smiles_sim.append(smils)
                long_smils = df_cluster.loc[count_df, 'long_smiles']
                long_smiles_sim.append(long_smils)
                count_smiles += 1
            count_df += 1
     
    new_mols = pd.DataFrame()        
    new_mols['smiles'] = smiles_sim
    
    return new_mols