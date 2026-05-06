import os
# 1. Fix the macOS OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 2. Standard Imports
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torchtuples as tt
from pycox.models import CoxPH

import sksurv.metrics
import sksurv.util
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.metrics import integrated_brier_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from DL_models.Genomic_Interaction_Encoder.MLP_SolidBiopsy import MLP_SolidBiopsy 
from sksurv.ensemble import RandomSurvivalForest
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)
simplefilter("ignore", category=RuntimeWarning)
simplefilter("ignore", category=FutureWarning)
  
def ML_model(method, sub_method, timepoints, df_pretraining, df_genes, df_list_genes, df_cln_vars, df_clinical_MSK,
               df_gene_ctDNA, df_panel_ctDNA, path_save_results_file):
    """
    Assess a Machine Learning model for survival analysis from longitudinal ctDNA data. 
    - Pre-train genomic-interaction encoder on MSK-CHORD dataset, then extract latent gene features 
    and use these as input to deep learning model for survival prediction.
    - Execute 100 repeats train-test on IMpower150 dataset for the given Deep Learning model 
    with selected architecture (Monte-Carlo Cross-Validation) 
    - Evaluate model performance on the 100 different test sets with C-index, AUC, and IBS.

    Args:
        method (str): choice of the machine learning model, one of 'RSF' (Random Survival Forest) 
        or 'penalized_cox' (penalized Cox proportional hazards regression)
        sub_method (str): one of 'ridge','elastic_net','lasso','ridge', or None if method is 'RSF' 
        timepoints (list): list of timepoints to include in model, e.g. ['BL','C2D1','C3D1','C4D1']
        df_pretraining (pd.DataFrame): pretraining dataset with mutation VAFs, e.g. MSK-CHORD dataset
        df_genes (pd.DataFrame): dataset with list of genes to include in the genomic-interaction encoder, e.g. set of genes commonly sequenced between MSK-CHORD and IMPower150 panels
        df_list_genes (pd.DataFrame): dataset with list of genes to include in the ctDNA-dynamics encoder, e.g. set of genes mutated in at least 5% of patients in IMPower150 dataset
        df_cln_vars (pd.DataFrame): dataset with tumor types in pretraining dataset, e.g. MSK-CHORD dataset
        df_clinical_MSK (pd.DataFrame): dataset with clinical variables and survival information for pretraining dataset, e.g. MSK-CHORD dataset
        df_gene_ctDNA (pd.DataFrame): dataset with longitudinal gene-level ctDNA features (Gene-level VAFs) for IMPower150 dataset
        df_panel_ctDNA (pd.DataFrame): dataset with longitudinal panel-level ctDNA features (e.g. Mean VAF, Max VAF, cfDNA conc., Nb driver mut.) for IMPower150 dataset
        path_save_results_file (str): path to save model performance metrics for each seed in a DataFrame
    
    Returns:
        Save model performance metrics (C-index, AUC, IBS) for each seed in a DataFrame. 
    """

    # Define the names of the time-to-event target variables for the pretraining dataset, 
    # which are overall survival time and event in the MSK-CHORD dataset.
    survival_time_name = 'OS_MONTHS'
    survival_event_name = 'OS_EVENT'
    # Store the number of timepoints
    d_channel = len(timepoints)
    df = df_pretraining
    # Select columns of interest and compute variant allele frequency, filter out unknown mutation status, and silent mutations
    columns_selected = ['Tumor_Sample_Barcode','Hugo_Symbol','Variant_Classification','Variant_Type','t_ref_count','t_alt_count','Mutation_Status']
    df = df.loc[:, columns_selected] 
    df['allele_frequency'] = df['t_alt_count'] / (df['t_alt_count'] + df['t_ref_count'])
    df = df[df['Mutation_Status'] != 'UNKNOWN']

    # Select genes that are commonly sequenced between MSK-CHORD and IMPower150 panels 
    list_genes = df_genes['gene'].tolist()
    list_genes_MSK = df['Hugo_Symbol'].unique().tolist()
    common_genes = list(set(list_genes_MSK) & set(list_genes))
    df = df[df['Hugo_Symbol'].isin(common_genes)]

    # Select the maximum allele frequency for each gene for each patient, 
    # and pivot to wide format with genes as columns and patients as rows
    df['PATIENT_ID'] = df['Tumor_Sample_Barcode'].str.split('-').str[:2].str.join('-')
    df = df[df['Variant_Classification'] != 'Silent']
    df = df[['PATIENT_ID','Hugo_Symbol','allele_frequency']]
    df = df.groupby(['PATIENT_ID', 'Hugo_Symbol'])['allele_frequency'].max().reset_index()
    df_wide = df.pivot(index='PATIENT_ID', columns='Hugo_Symbol', values='allele_frequency')
    df_wide = df_wide.fillna(0)
    df_wide = df_wide.reset_index()
    df_wide = df_wide.rename_axis(None, axis=1)
    df = df_wide

    # Remove genes that are mutated in less than 2% of patients, as these are unlikely 
    # to be informative for survival prediction and may add noise to the model
    for col in df.columns:
        if sum(df[col]==0)>0.98*len(df):
            df = df.drop(columns=[col])

    # Map clinical variable of highest stage recorded to binary variable of metastatic 
    # (stage 4) vs non-metastatic (stage 1-3), and merge with genomic data
    df_clinical_MSK['Highest_Stage'] = None
    df_clinical_MSK.loc[df_clinical_MSK['STAGE_HIGHEST_RECORDED']=='Stage 1-3','Highest_Stage']=0
    df_clinical_MSK.loc[df_clinical_MSK['STAGE_HIGHEST_RECORDED']=='Stage 4','Highest_Stage']=1
    df_clinical_MSK = df_clinical_MSK[df_clinical_MSK['STAGE_HIGHEST_RECORDED']!='Unknown']
    df_clinical_MSK = df_clinical_MSK[['PATIENT_ID', 'Highest_Stage', 'OS_MONTHS', 'OS_STATUS']]
    df = df.merge(df_clinical_MSK, left_on='PATIENT_ID', right_on='PATIENT_ID', how='inner')

    # One-hot encode tumor types and merge with genomic and stage data, and filter to patients with available survival information
    df_dummies = pd.get_dummies(df_cln_vars, columns=['CANCER_TYPE'], prefix='')
    df_dummies.rename(columns={'_Breast Cancer': 'Breast', '_Colorectal Cancer': 'Colorectal', '_Non-Small Cell Lung Cancer': 'NSCLC', '_Pancreatic Cancer': 'Pancreatic', '_Prostate Cancer':'Prostate'}, inplace=True)
    df_dummies = df_dummies.groupby('PATIENT_ID').agg(
        Breast=('Breast', 'sum'),
        Colorectal=('Colorectal', 'sum'),
        NSCLC=('NSCLC', 'sum'),
        Pancreatic=('Pancreatic', 'sum'),
        Prostate=('Prostate', 'sum'), 
        TMB=('TMB_NONSYNONYMOUS', 'max'))

    df = df.merge(df_dummies, left_on='PATIENT_ID', right_on='PATIENT_ID', how='inner')
    df = df[df['OS_MONTHS'].notna()]

    # Map overall survival status to binary variable of deceased vs living, and 
    # drop original OS_STATUS variable and patient ID variable
    df['OS_EVENT'] = df['OS_STATUS'].map({'1:DECEASED': 1,'0:LIVING': 0})
    df['OS_EVENT'] = df['OS_EVENT'].astype('int')
    df = df.drop(columns=['OS_STATUS', 'PATIENT_ID'])    
    
    # Select gene features and clinical features for model input, 
    # and rename survival columns for consistency with later code
    df_full_train = df       
    gene_and_cln_names = [i for i in df_full_train.columns if 'OS_' not in i]
    gene_names = [i for i in gene_and_cln_names if 'Highest_Stage' not in i and 'Breast' not in i and 'Colorectal' not in i and 'NSCLC' not in i and 'Pancreatic' not in i and 'Prostate' not in i and 'TMB' not in i] 
    cln_names = ['Highest_Stage','Breast','Colorectal','NSCLC','Pancreatic','Prostate','TMB']    
    genes_and_survival_names = gene_names + ['OS_MONTHS','OS_EVENT']
    df_full_train_genes = df_full_train[genes_and_survival_names].copy()
    df_full_train_genes = df_full_train_genes.rename(columns={'OS_MONTHS': 'OS_rebaseline', 'OS_EVENT': 'OS_event'})


    class molecular_survival_BigDataset(Dataset):
        
        """
        PyTorch Dataset class to load gene features, clinical features, and survival information for each patient,
        and return as tensors for model input. This will be used for pretraining the genomic-interaction encoder on the MSK-CHORD dataset,
        and for extracting latent gene features for the main survival prediction model on the IMpower150 dataset.
        """
        
        def __init__(self, dataset_mol, survival_time_name='OS_MONTHS', survival_event_name='OS_EVENT'):
            self.dataset_mol = dataset_mol
            self.survival_time_name = survival_time_name
            self.survival_event_name = survival_event_name
            
        def __len__(self):
            return len(self.dataset_mol)
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
                
            gene_features = self.dataset_mol.iloc[idx,][gene_names].to_numpy().astype(np.float32).reshape((1,-1))
            gene_features = torch.from_numpy(gene_features)
            cln_features = self.dataset_mol.iloc[idx,][cln_names].to_numpy().astype(np.float32).reshape((1,-1))
            cln_features = torch.from_numpy(cln_features)
            targets = (self.dataset_mol[self.survival_time_name].iloc[idx].astype(np.float32), self.dataset_mol[self.survival_event_name].iloc[idx])
                
            return gene_features, targets, cln_features
        

    # Set random seeds for reproducibility, and define function to extract survival time 
    # and event information from dataset 
    np.random.seed(1)
    _ = torch.manual_seed(1)
    get_target = lambda df: (df[survival_time_name].values, df[survival_event_name].values)

    # Split dataset into training and test sets, and further split training set into 
    # training and validation sets for pretraining the genomic-interaction encoder on the MSK-CHORD dataset.
    df_vaf = df.copy()            
    df_vaf_full_train = df_vaf
    df_vaf_val = df_vaf.sample(frac=0.4)
    df_vaf_train = df_vaf.drop(df_vaf_val.index)    
        
    # Create PyTorch Datasets and DataLoaders for training and validation sets for pretraining the genomic-interaction encoder on the MSK-CHORD dataset.                
    train_dataset = molecular_survival_BigDataset(df_vaf_train)
    trainloader = DataLoader(train_dataset, batch_size=df_vaf_train.shape[0], shuffle=False)    
    val_dataset =  molecular_survival_BigDataset(df_vaf_val)
    valloader = DataLoader(val_dataset, batch_size=df_vaf_val.shape[0], shuffle=False)
           
    # Define the architecture and hyperparameters for the genomic-interaction encoder             
    d_input_SB = len(gene_names)
    d_cln_SB = len(cln_names)
    d_channel_SB = 1
    d_output_SB = 1  
    dropout_SB = 0.5        
    DEVICE_SB = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    net_SB = MLP_SolidBiopsy(d_input=d_input_SB, d_channel=d_channel_SB, d_output=d_output_SB, d_other_input=d_cln_SB,
                    device=DEVICE_SB, dropout=dropout_SB)    
    model = CoxPH(net_SB, tt.optim.Adam)
    batch_size = 500
    model.optimizer.set_lr(0.01)
    callbacks = [tt.callbacks.EarlyStopping(patience=100)]
    verbose = False
    epochs = 512
          
    # Extract gene features, clinical features, and survival information from training and validation dataloaders         
    for i, data in enumerate(trainloader, 0):
        gene_features, targets, cln_features = data
    for l, vdata in enumerate(valloader, 0):
        val_gene_features, val_targets, val_cln_features = vdata
        val = tt.tuplefy([val_gene_features.float(), val_cln_features.float()], val_targets)

    # Pretrain the encoder on the MSK-CHORD dataset to learn latent gene features 
    # that capture prognostic interactions between mutated genes.

    log = model.fit([gene_features.float(), cln_features.float()], targets, batch_size, epochs, callbacks, verbose,val_data=val.repeat(10).cat())        

    # Define function to extract latent gene features from the pretrained encoder 
    # in order to use them in the full Deep Learning for survival analysis in the IMpower150 cohort
    def latent_gene_features(dataset):        
        latent_gene_encoding_all = []
        for timepoint in timepoints:   
            latent_gene_encoding = torch.from_numpy(dataset[[col + '_' + timepoint for col in gene_names]].to_numpy().astype(np.float32))
            latent_gene_encoding = net_SB.extract_encoder(latent_gene_encoding).detach().cpu()
            latent_gene_encoding_all.append(latent_gene_encoding)
            return torch.stack(latent_gene_encoding_all, dim=-1)
    
    # Extract the list of genes to include in the ctDNA-dynamics encoder, 
    # which are those mutated in at least 5% of patients in the IMPower150 dataset    
    gene_list_restricted = df_list_genes['gene_kept_restricted'].unique().tolist()
    
    # Define the names of time-to-event target for IMpower150 dataset
    survival_time_name='OS_rebaseline' 
    survival_event_name='OS_event' 
    survival_names = ['OS_rebaseline','OS_event']
    
    # Select the relevant features for the main survival prediction model on the IMpower150 dataset, 
    # including the longitudinal gene-level VAF features, the longitudinal panel-level aggregated ctDNA features, 
    # and the clinical features.
    features_list = ['_max_of_AF','_mean_of_AF','_cfDNA_conc','_n_kl']
    agg_features_list = []
    for timepoint in timepoints:
        agg_features_list += ["".join((timepoint,features_list[i])) for i in range(len(features_list))]

    # Select the relevant features for the main survival prediction model on the IMpower150 dataset, 
    # including the longitudinal gene-level VAF features, the longitudinal panel-level aggregated ctDNA features, 
    # and the clinical features.
    Long_Agg = ['full_anonymized_patient_id']+agg_features_list
    df_panel_ctDNA = df_panel_ctDNA[Long_Agg]
    df_gene_ctDNA = df_gene_ctDNA.drop(columns=[i for i in df_gene_ctDNA.columns if 'Agg' in i])
    df_gene_ctDNA = df_gene_ctDNA.reset_index(drop=False).rename(columns={'index': 'full_anonymized_patient_id'})
    df_1 = pd.merge(df_gene_ctDNA,df_panel_ctDNA,left_on='full_anonymized_patient_id',right_on='full_anonymized_patient_id',how='inner')
    df_1 = df_1.drop(columns=['full_anonymized_patient_id'])
    cols_mol = [i for i in df_1.columns if 'Agg' not in i and 'cln' not in i and 'OS_event' not in i and 'OS_rebaseline' not in i and 'PFS' not in i]
    cols_mol_genes = [i for i in cols_mol if '_BL' in i or '_C2D1' in i or '_C3D1' in i or '_C4D1' in i]
    list_feat = agg_features_list.copy()
    list_feat += [i for i in df_1.columns if 'cln_' in i or survival_time_name in i or survival_event_name in i]
    list_feat += cols_mol_genes.copy()  
    df_1 = df_1[list_feat]
    
    # Initialize lists to store model performance metrics 
    # across different seeds for Monte-Carlo Cross-Validation
    C_INDEX = []
    AUC = []
    IBS = []
    # Transform the target variable for survival analysis as a tuple of (time, event) 
    # for use in the PyTorch Datasets and model training
    get_target = lambda df: (df[survival_time_name].values, df[survival_event_name].values)

    # Run Monte-Carlo Cross-Validation with 100 different random seeds 
    # to split the IMPower150 dataset into training and test sets, 
    # train the deep learning model on the training set, and evaluate performance on the test set for each seed.
    for seed in tqdm(range(0, 100)):
        
        # Set random seeds for reproducibility for each iteration of Monte-Carlo Cross-Validation
        np.random.seed(1*seed)
        _ = torch.manual_seed(1*seed)
        
        # Split dataset into training and test sets, and further split training set into 
        # training and validation sets for training the main survival prediction model on the IMpower150 dataset.
        df_vaf=df_1.copy()        
        df_vaf_test = df_vaf.sample(frac=0.3)
        df_vaf = df_vaf.drop(df_vaf_test.index)
        df_vaf_full_train = df_vaf
        
        # Extract latent gene features from the pretrained genomic-interaction encoder 
        # for each patient in the training, validation, and test sets of the IMPower150 dataset, 
        # to use as input to the main survival prediction model.
        lat_gene_features_test = latent_gene_features(dataset=df_vaf_test)
        lat_gene_features_fulltrain = latent_gene_features(dataset=df_vaf_full_train)
               
        # Extract survival time and event information from the training and test sets of the IMPower150 dataset,
        # and format as structured array for use in model training and evaluation with sksurv metrics 
        durations_test, events_test = get_target(df_vaf_test)             
        durations_train_full, events_train_full = get_target(df_vaf_full_train)
        y_full_train_formatted = sksurv.util.Surv.from_arrays(event=events_train_full, time=durations_train_full)
        durations_test, events_test = get_target(df_vaf_test)
        y_test_formatted = sksurv.util.Surv.from_arrays(event=events_test, time=durations_test)
            
        # Select the gene-level ctDNA features, the panel-level aggregated ctDNA features, and the clinical features 
        # for the main survival prediction model on the IMPower150 dataset, and concatenate with the latent gene features 
        # extracted from the pretrained genomic-interaction encoder.
        cols_mol_genes_restricted = []
        for timepoint in timepoints:
            cols_mol_genes_restricted += ['_'.join((gene_name,timepoint)) for gene_name in gene_list_restricted]
        features_selected = cols_mol_genes_restricted + agg_features_list 
        df_vaf_full_train = df_vaf_full_train[features_selected+survival_names]
        df_vaf_test = df_vaf_test[features_selected+survival_names]
        df_vaf_full_train = df_vaf_full_train.reset_index(drop=True)
        df_vaf_full_train = pd.concat([df_vaf_full_train, pd.DataFrame(lat_gene_features_fulltrain.reshape(lat_gene_features_fulltrain.shape[0],-1))], axis=1)
        df_vaf_full_train.columns = df_vaf_full_train.columns.map(str)
        df_vaf_test = df_vaf_test.reset_index(drop=True)    
        df_vaf_test = pd.concat([df_vaf_test, pd.DataFrame(lat_gene_features_test.reshape(lat_gene_features_test.shape[0],-1))], axis=1)    
        df_vaf_test.columns = df_vaf_test.columns.map(str)
        features = [i for i in df_vaf_full_train.columns if survival_time_name not in i and survival_event_name not in i]        
        cv = KFold(n_splits=4, shuffle=True, random_state=1)
         
        # Train the selected machine learning model (Random Survival Forest or penalized Cox regression) 
        # on the training set of the IMPower150 dataset 
        if method=='penalized_cox':
            if sub_method=='elastic_net':
                l1_ratio = 0.5
            elif sub_method=='lasso':
                l1_ratio = 1.0
            elif sub_method=='ridge':
                l1_ratio = 1e-5
                
            estimator = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=20, max_iter=5000, alpha_min_ratio=0.005)
            estimator.fit(df_vaf_full_train[features], y_full_train_formatted)
            estimated_alphas = estimator.alphas_
            gcv = GridSearchCV(CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=5000, fit_baseline_model=True),
                param_grid={"alphas": [[v] for v in estimated_alphas]},
                cv=cv,
                error_score=0.5,
                n_jobs=1,
                ).fit(df_vaf_full_train[features], y_full_train_formatted)

        elif method=='RSF':
            gcv = GridSearchCV(RandomSurvivalForest(),
                param_grid={"max_depth": [15, None],
                            "min_samples_split": [3, 6]},
                cv=cv,
                error_score=0.5,
                n_jobs=4,
                ).fit(df_vaf_full_train[features], y_full_train_formatted)

        best_model = gcv.best_estimator_

        # Evaluate model performance on the test set of the IMPower150 dataset with C-index, AUC, and IBS 
        # metrics, and store results for each seed in lists to compute mean and standard deviation across 
        # seeds at the end. 
        c_index = sksurv.metrics.concordance_index_ipcw(survival_train=y_full_train_formatted,survival_test=y_test_formatted,estimate=best_model.predict(df_vaf_test[features]))[0]
        C_INDEX = np.append(C_INDEX,c_index)
        print('C-index',c_index)
        
        times = np.arange(2.5,40)
        if d_channel==2:
            times = np.arange(2.55,40)
        auc = sksurv.metrics.cumulative_dynamic_auc(survival_train=y_full_train_formatted,survival_test=y_test_formatted,estimate=best_model.predict(df_vaf_test[features]), times=times)[1]
        AUC = np.append(AUC, auc)
        print('AUC',auc)

        survs = best_model.predict_survival_function(df_vaf_test[features])
        base_times = survs[0].x
        times = base_times[(base_times >= 2.1) & (base_times <= 40)]
        if d_channel==2:
            times = base_times[(base_times >= 2.55) & (base_times <= 40)]
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
        ibs = integrated_brier_score(survival_train=y_full_train_formatted,survival_test=y_test_formatted,estimate=preds, times=times)
        print('IBS',ibs)
        IBS = np.append(IBS, ibs)

    print('C-index mean/std:', C_INDEX.mean(), C_INDEX.std())
    print('AUC mean/std:', AUC.mean(), AUC.std())
    print('IBS Sksurv mean/std:', IBS.mean(), IBS.std())

    df_results = pd.DataFrame({
        "C-index": C_INDEX,
        "AUC": AUC,
        "IBS": IBS
    })

    print(df_results)
    df_results.to_csv(path_save_results_file, index=True)
    
# Execute the function with the selected machine learning model architecture 
# and the relevant datasets for pretraining and survival prediction, and save results to a file.

timepoints = ['BL','C2D1','C3D1','C4D1']
d_channel = len(timepoints)
method='penalized_cox' #'RSF','penalized_cox'
sub_method='ridge' #'elastic_net','lasso','ridge','' 
df_pretraining = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_mutations.txt', sep='\t')
df_genes = pd.read_csv(''.join(("/Users/aaronmamann/Documents/ImPower150_Genes_",str(d_channel),".csv")),index_col=0)
df_cln_vars = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_clinical_sample.txt', sep='\t', skiprows=4)
df_clinical_MSK = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_clinical_patient.txt', sep='\t', skiprows=4)
df_list_genes = pd.read_csv(''.join(("/Users/aaronmamann/Documents/ImPower150_geneskept_restricted_",str(d_channel),".csv")),index_col=0)
df_gene_ctDNA = pd.read_csv("".join(("/Users/aaronmamann/Documents/ImPower150_OS_",str(d_channel),".csv")),index_col=0) 
df_panel_ctDNA = pd.read_csv("".join(("/Users/aaronmamann/Documents/ImPower150_AggFeatures_",str(d_channel),".csv")),index_col=0)
path_save_results_file = ''.join(('Cycles_exp/',method,sub_method,'_metrics_',str(d_channel),'.csv'))
ML_model(method=method, sub_method=sub_method, timepoints=timepoints, df_pretraining=df_pretraining, 
         df_genes=df_genes, df_list_genes=df_list_genes, df_cln_vars=df_cln_vars, df_clinical_MSK=df_clinical_MSK,
         df_gene_ctDNA=df_gene_ctDNA, df_panel_ctDNA=df_panel_ctDNA, path_save_results_file=path_save_results_file)
