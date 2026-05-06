import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torchtuples as tt
from pycox.models import CoxPH
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from DL_models.Genomic_Interaction_Encoder.MLP_SolidBiopsy import MLP_SolidBiopsy 
from DL_models.SHERLOC import SHERLOC
from utils import calculate_corrected_confidence_intervals
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)
simplefilter("ignore", category=RuntimeWarning)
simplefilter("ignore", category=FutureWarning)


def explainability(timepoints, df_pretraining, df_genes, df_cln_vars, df_clinical_MSK, 
                   df_list_genes, df_gene_ctDNA, df_panel_ctDNA, path_save_results_file):
    
    """
    Explainability of a Deep Learning model for survival analysis from longitudinal ctDNA data. 
    - Pre-train genomic-interaction encoder on MSK-CHORD dataset, then extract latent gene features 
    and use these as input to deep learning model for survival prediction.
    - Train SHERLOC on IMpower150 dataset  
    - Freeze the longitudinal representation layers and select a bootstrap sample from training set, 
    re-initialize the output layer, retrain the output layer on the bootstrap sample, and save the output layer weights. 
    Repeat this process for 100 bootstrap samples, and obtain confidence intervals for the weights of longitudinal 
    representations of each features (gene- and panel-level ctDNA features) using the correction of Nadeau and Bengio 
    to account for dependence between bootstrap samples. Perform a resampled t-test following Nadeau and Bengio methodology 
    to assess the statistical significance of each feature's contribution to the model's predictions.

    Args:
        timepoints (list): list of timepoints to include in model, e.g. ['BL','C2D1','C3D1','C4D1']
        df_pretraining (pd.DataFrame): pretraining dataset with mutation VAFs, e.g. MSK-CHORD dataset
        df_genes (pd.DataFrame): dataset with list of genes to include in the genomic-interaction encoder, e.g. set of genes commonly sequenced between MSK-CHORD and IMPower150 panels
        df_list_genes (pd.DataFrame): dataset with list of genes to include in the ctDNA-dynamics encoder, e.g. set of genes mutated in at least 5% of patients in IMPower150 dataset
        df_cln_vars (pd.DataFrame): dataset with tumor types in pretraining dataset, e.g. MSK-CHORD dataset
        df_clinical_MSK (pd.DataFrame): dataset with clinical variables and survival information for pretraining dataset, e.g. MSK-CHORD dataset
        df_gene_ctDNA (pd.DataFrame): dataset with longitudinal gene-level ctDNA features (Gene-level VAFs) for IMPower150 dataset
        df_panel_ctDNA (pd.DataFrame): dataset with longitudinal panel-level ctDNA features (e.g. Mean VAF, Max VAF, cfDNA conc., Nb driver mut.) for IMPower150 dataset
        path_save_results_file (str): path to save model weights, confidence intervals, and p-values for longitudinal representations of each feature.
    
    Returns:
        Save model weights, confidence intervals, and p-values for longitudinal representations of each feature in a DataFrame. 
    """    
    
    df = df_pretraining
    d_channel = len(timepoints) 
    
    # Select relevant columns and compute allele frequency
    columns_selected = ['Tumor_Sample_Barcode','Hugo_Symbol','Variant_Classification','Variant_Type','t_ref_count','t_alt_count','Mutation_Status']
    df = df.loc[:, columns_selected] 
    df['allele_frequency'] = df['t_alt_count'] / (df['t_alt_count'] + df['t_ref_count'])
    
    # Filter out unknown mutation status and genes not in the list of genes to include in the 
    # genomic-interaction encoder 
    df = df[df['Mutation_Status'] != 'UNKNOWN']
    list_genes = df_genes['gene'].tolist()
    list_genes_MSK = df['Hugo_Symbol'].unique().tolist()
    common_genes = list(set(list_genes_MSK) & set(list_genes))
    df = df[df['Hugo_Symbol'].isin(common_genes)]
    df['PATIENT_ID'] = df['Tumor_Sample_Barcode'].str.split('-').str[:2].str.join('-')
    
    # Filter out silent mutations and keep the maximum allele frequency for each gene for each patient
    df = df[df['Variant_Classification'] != 'Silent']
    df = df[['PATIENT_ID','Hugo_Symbol','allele_frequency']]
    df = df.groupby(['PATIENT_ID', 'Hugo_Symbol'])['allele_frequency'].max().reset_index()
    df_wide = df.pivot(index='PATIENT_ID', columns='Hugo_Symbol', values='allele_frequency')
    df_wide = df_wide.fillna(0)
    df_wide = df_wide.reset_index()
    df_wide = df_wide.rename_axis(None, axis=1)
    df = df_wide
    
    # Filter out genes that are mutated in less than 2% of patients in the pretraining dataset
    for col in df.columns:
        if sum(df[col]==0)>0.98*len(df):
            df = df.drop(columns=[col])

    # Merge with clinical dataset to obtain survival information and tumor types for pretraining dataset
    df_clinical_MSK['Highest_Stage'] = None
    df_clinical_MSK.loc[df_clinical_MSK['STAGE_HIGHEST_RECORDED']=='Stage 1-3','Highest_Stage']=0
    df_clinical_MSK.loc[df_clinical_MSK['STAGE_HIGHEST_RECORDED']=='Stage 4','Highest_Stage']=1
    df_clinical_MSK = df_clinical_MSK[df_clinical_MSK['STAGE_HIGHEST_RECORDED']!='Unknown']
    df_clinical_MSK = df_clinical_MSK[['PATIENT_ID', 'Highest_Stage', 'OS_MONTHS', 'OS_STATUS']]
    df = df.merge(df_clinical_MSK, left_on='PATIENT_ID', right_on='PATIENT_ID', how='inner')
    
    # Create dummy variables for tumor types and keep the maximum TMB for each patient
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
    df['OS_EVENT'] = df['OS_STATUS'].map({
        '1:DECEASED': 1,
        '0:LIVING': 0
    })
    df['OS_EVENT'] = df['OS_EVENT'].astype('int')
    df = df.drop(columns=['OS_STATUS', 'PATIENT_ID'])
    
    # Define survival time and event columns, and function to extract these as target variables for survival analysis
    survival_time_name = 'OS_MONTHS'
    survival_event_name = 'OS_EVENT'
    df_full_train = df   
    
    # Define gene names and clinical variable names, and create dataset with gene features and 
    # survival information for pretraining dataset
    gene_and_cln_names = [i for i in df_full_train.columns if 'OS_' not in i]
    gene_names = [i for i in gene_and_cln_names if 'Highest_Stage' not in i and 'Breast' not in i and 'Colorectal' not in i and 'NSCLC' not in i and 'Pancreatic' not in i and 'Prostate' not in i and 'TMB' not in i] 
    cln_names = ['Highest_Stage','Breast','Colorectal','NSCLC','Pancreatic','Prostate','TMB']
    genes_and_survival_names = gene_names + ['OS_MONTHS','OS_EVENT']
    df_full_train_genes = df_full_train[genes_and_survival_names].copy()
    df_full_train_genes = df_full_train_genes.rename(columns={'OS_MONTHS': 'OS_rebaseline', 'OS_EVENT': 'OS_event'})

    # Define PyTorch dataset class for pretraining dataset
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
        

    # Set random seeds for reproducibility
    np.random.seed(1)
    _ = torch.manual_seed(1)
    df_vaf = df.copy()
    
    # Split pretraining dataset into training and validation sets     
    df_vaf_full_train = df_vaf
    df_vaf_val = df_vaf.sample(frac=0.4)
    df_vaf_train = df_vaf.drop(df_vaf_val.index)    
    
    # Create PyTorch dataloaders for training and validation sets
    train_dataset = molecular_survival_BigDataset(df_vaf_train)
    trainloader = DataLoader(train_dataset, batch_size=df_vaf_train.shape[0], shuffle=False)
    val_dataset =  molecular_survival_BigDataset(df_vaf_val)
    valloader = DataLoader(val_dataset, batch_size=df_vaf_val.shape[0], shuffle=False)
        
    # Define model architecture and training parameters for pretraining on MSK-CHORD dataset
    d_input_SB = len(gene_names)
    d_cln_SB = len(cln_names)
    d_channel_SB = 1
    d_output_SB = 1  
    dropout_SB = 0.5  
    DEVICE_SB = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    # Initialize the genomic-interaction encoder and survival model, and set training parameters
    net_SB = MLP_SolidBiopsy(d_input=d_input_SB, d_channel=d_channel_SB, d_output=d_output_SB, d_other_input=d_cln_SB,
                    device=DEVICE_SB, dropout=dropout_SB)
    model = CoxPH(net_SB, tt.optim.Adam)
    batch_size = 500
    epochs = 512
    model.optimizer.set_lr(0.01)
    callbacks = [tt.callbacks.EarlyStopping(patience=100)]
    verbose = False
    
    # Extract gene features, clinical features, and target variables from training and validation dataloaders     
    for i, data in enumerate(trainloader, 0):
        gene_features, targets, cln_features = data          
    for l, vdata in enumerate(valloader, 0):
        val_gene_features, val_targets, val_cln_features = vdata
        val = tt.tuplefy([val_gene_features.float(), val_cln_features.float()], val_targets)
    
    # Train the model on the training set and evaluate on the validation set, using the 
    # extracted gene features, clinical features, and target variables
    log = model.fit([gene_features.float(), cln_features.float()], targets, batch_size, epochs, callbacks, verbose,val_data=val.repeat(10).cat())        
    
    # Extract the latent gene features from the pretrained genomic-interaction encoder for the IMPower150 dataset, 
    def latent_gene_features(dataset):
        
            latent_gene_encoding_all = []
            for timepoint in timepoints:   
                latent_gene_encoding = torch.from_numpy(dataset[[col + '_' + timepoint for col in gene_names]].to_numpy().astype(np.float32))
                latent_gene_encoding = net_SB.extract_encoder(latent_gene_encoding).detach().cpu()
                latent_gene_encoding_all.append(latent_gene_encoding)
        
            return torch.stack(latent_gene_encoding_all, dim=-1)
    
    # Extract the list of genes to include in the ctDNA-dynamics encoder for the IMPower150 dataset
    gene_list_restricted = df_list_genes['gene_kept_restricted'].unique().tolist()
    
    # Define PyTorch dataset class for IMPower150 dataset, which includes gene features extracted 
    # from the pretrained genomic-interaction encoder, panel-level ctDNA features, clinical features, 
    # and survival information for each patient    
    class molecular_survival(Dataset):
        
            """
            PyTorch Dataset class to load gene features, clinical features, 
            aggregated ctDNA features, and survival information for each patient,
            and return as tensors for model input. This will be used for training 
            the main survival deep learning model on the IMpower150 dataset, 
            with the pretrained latent gene features as input to the genomic-interaction encoder.
            """        
            
            def __init__(self, dataset_vaf, survival_event_name, survival_time_name, tensor_gene_encoded):
                self.dataset_vaf = dataset_vaf
                self.survival_event_name=survival_event_name
                self.survival_time_name=survival_time_name
                self.tensor_gene_encoded = tensor_gene_encoded 
                
            def __len__(self):
                return len(self.dataset_vaf)
            
            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                                
                cols_mol_agg = ['max_of_AF','mean_of_AF','cfDNA_conc','n_kl']
                cols_mol_genes = gene_list_restricted 

                patient_features = np.ones((len(cols_mol_genes),d_channel))
                patient_agg_features = np.ones((len(cols_mol_agg),d_channel))

                i=0                    
                for feature in cols_mol_genes:
                    feature_timesteps = ["_".join((feature,timepoint)) for timepoint in timepoints]
                    patient_features[i]=self.dataset_vaf.iloc[idx,][feature_timesteps].to_numpy().astype(np.float32).reshape((1,-1))
                    i=i+1
                
                i=0    
                for feature in cols_mol_agg:
                    feature_timesteps = ["_".join((timepoint,feature)) for timepoint in timepoints]
                    patient_agg_features[i]=self.dataset_vaf.iloc[idx,][feature_timesteps].to_numpy().astype(np.float32).reshape((1,-1))
                    i=i+1
                    
                patient_features = torch.from_numpy(patient_features)
                patient_agg_features = torch.from_numpy(patient_agg_features) 
                
                patient_survival = (self.dataset_vaf[self.survival_time_name].iloc[idx].astype(np.float32), self.dataset_vaf[self.survival_event_name].iloc[idx])
                
                patient_clinical = ['cln_pdl1_high','cln_nb_mets','cln_baseline_ECOG','cln_age','cln_sex_female','cln_sld_baseline','cln_ABCP','cln_BCP']

                patient_cln = self.dataset_vaf.iloc[idx,][patient_clinical].to_numpy().astype(np.float32).reshape((1,-1))
                patient_cln = torch.from_numpy(patient_cln)
                
                return patient_features, patient_survival, patient_cln, patient_agg_features, self.tensor_gene_encoded[idx].float() 


    # Define the names of time-to-event target for IMpower150 dataset    
    survival_time_name='OS_rebaseline' 
    survival_event_name='OS_event' 
    
    # Select the relevant features for the main survival prediction model on the IMpower150 dataset, 
    # including the longitudinal gene-level VAF features, the longitudinal panel-level aggregated ctDNA features, 
    # and the clinical features.
    features_list = ['_max_of_AF','_mean_of_AF','_cfDNA_conc','_n_kl']
    agg_features_list = []
    for timepoint in timepoints:
        agg_features_list += ["".join((timepoint,features_list[i])) for i in range(len(features_list))]
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
    
    # Transform the target variable for survival analysis as a tuple of (time, event) 
    # for use in the PyTorch Datasets and model training
    get_target = lambda df: (df[survival_time_name].values, df[survival_event_name].values)
    
    # Set random seeds for reproducibility for interpretation of SHERLOC on the IMpower150 dataset    
    np.random.seed(16)
    _ = torch.manual_seed(16)
    df_vaf=df_1.copy()
    
    # Split the IMPower150 dataset into training, validation, and test sets, and create 
    # PyTorch dataloaders for each set.       
    df_vaf_test = df_vaf.sample(frac=0.3, random_state=16)
    df_vaf = df_vaf.drop(df_vaf_test.index)
    df_vaf_full_train = df_vaf
    df_vaf_val = df_vaf.sample(frac=0.2)
    df_vaf_train = df_vaf.drop(df_vaf_val.index)
    
    # Extract the latent gene features from the pretrained genomic-interaction encoder for 
    # the training and validation sets of the IMPower150 dataset, to use as input to 
    # the genomic-interaction encoder in the main survival prediction model.
    lat_gene_features_train = latent_gene_features(dataset=df_vaf_train)
    lat_gene_features_val = latent_gene_features(dataset=df_vaf_val)

    # Create PyTorch dataloaders for training and validation sets of the IMPower150 dataset,
    # using the extracted latent gene features as input to the genomic-interaction encoder 
    # in the main survival prediction model. The dataloader will return the longitudinal gene 
    # features, the longitudinal panel-level aggregated ctDNA features, the clinical features, 
    # and the survival information for each patient in the training and validation sets.
    train_dataset = molecular_survival(dataset_vaf=df_vaf_train, 
                                            survival_event_name=survival_event_name, 
                                            survival_time_name=survival_time_name,
                                            tensor_gene_encoded = lat_gene_features_train)
    trainloader = DataLoader(train_dataset, batch_size=len(df_vaf_train), shuffle=False)
    val_dataset =  molecular_survival(dataset_vaf=df_vaf_val, 
                                            survival_event_name=survival_event_name, 
                                            survival_time_name=survival_time_name,
                                            tensor_gene_encoded = lat_gene_features_val)
    valloader = DataLoader(val_dataset, batch_size=len(df_vaf_val), shuffle=False)
                
    # Define the model architecture and training parameters for the main survival prediction model 
    d_input = train_dataset[0][0].shape[0]
    d_other_input = train_dataset[0][2].shape[1]
    d_agg_input = train_dataset[0][3].shape[0]
    d_output = 1  
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    # Initialize the SHERLOC model for survival prediction on the IMPower150 dataset           
    net = SHERLOC(d_input=d_input, d_agg_input=d_agg_input, d_other_input=d_other_input, d_channel=d_channel, 
                            d_output=d_output, device=DEVICE)             
    model = CoxPH(net, tt.optim.Adam)    
    batch_size = 50
    model.optimizer.set_lr(0.005)
    epochs = 256
    callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    verbose = False
    
    # Extract the gene features, clinical features, panel-level ctDNA features, and target variables from the training and 
    # validation dataloaders for the IMPower150 dataset, to use as input for training the SHERLOC model.    
    for i, data in enumerate(trainloader, 0):
                inputs, targets, other_inputs, agg_inputs, gene_encoded_inputs = data
    for l, vdata in enumerate(valloader, 0):
                vinputs, vtargets, vother_inputs, vagg_inputs, vgene_encoded_inputs = vdata
                val = tt.tuplefy([vinputs.float(), vagg_inputs.float(),vother_inputs.float(), vgene_encoded_inputs.float()], vtargets)

    # Train the SHERLOC model on the training set of the IMPower150 dataset
    log = model.fit([inputs.float(), agg_inputs.float(),other_inputs.float(), gene_encoded_inputs.float()], targets, batch_size, epochs, callbacks, verbose,val_data=val.repeat(10).cat())        
    
    df_weight_time = pd.DataFrame(model.net.filter1.weight.detach().cpu().numpy())
    df_weights = pd.DataFrame(model.net.fc_1.weight.detach().cpu().numpy().flatten())

    # Perform bootstrap sampling and retraining of the output layer to obtain confidence intervals 
    # for the weights of the longitudinal representations of each feature, using the correction of Nadeau and Bengio to 
    # account for dependence between bootstrap samples.
    weight_time_models = [] 
    weights_models = []      
    for random_state in tqdm(range(100)):
            
            # Bootstrap sampling of the training set, with replacement, to create a new training set for 
            # retraining the output layer of the SHERLOC model.
            df_sub_train = df_vaf_full_train.sample(frac=0.8, replace=False, random_state=10*random_state)
            df_sub_val = df_vaf_full_train.drop(df_sub_train.index)
            lat_gene_features_sub_train = latent_gene_features(dataset=df_sub_train)
            lat_gene_features_sub_val = latent_gene_features(dataset=df_sub_val)
            
            # Create PyTorch dataloaders for the bootstrap training and validation sets
            sub_train_dataset =  molecular_survival(dataset_vaf=df_sub_train, 
                                                    survival_event_name=survival_event_name, 
                                                    survival_time_name=survival_time_name, 
                                                    tensor_gene_encoded = lat_gene_features_sub_train)
            subtrainloader = DataLoader(sub_train_dataset, batch_size=len(df_sub_train), shuffle=False)
            sub_val_dataset =  molecular_survival(dataset_vaf=df_sub_val, 
                                                    survival_event_name=survival_event_name, 
                                                    survival_time_name=survival_time_name, 
                                                    tensor_gene_encoded = lat_gene_features_sub_val)
            subvalloader = DataLoader(sub_val_dataset, batch_size=len(df_sub_val), shuffle=False)
            # Initialize a new SHERLOC model and load the pretrained weights from the original model, 
            # then re-initialize the output layer and retrain on the bootstrap sample.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            modelA = SHERLOC(d_input=d_input, d_agg_input=d_agg_input, d_other_input=d_other_input, d_channel=d_channel, 
                            d_output=d_output, device=DEVICE).to(device)
            modelA.load_state_dict(model.net.state_dict()) 
            
            # re-intialize output layer
            nn.init.xavier_uniform_(modelA.fc_1.weight)
            nn.init.zeros_(modelA.fc_1.bias)
            
            # Freeze everything
            for p in modelA.parameters():
                p.requires_grad = False

            # Unfreeze output layer
            for p in modelA.fc_1.parameters():
                p.requires_grad = True

            # Define optimizer for retraining the output layer on the bootstrap sample
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, modelA.parameters()),
                lr=0.005
            )
            
            model_bt = CoxPH(modelA, optimizer)            
            batch_size = 50
            callbacks = [tt.callbacks.EarlyStopping(patience=20)]
            verbose = False
            epochs = 256

            # Extract the gene features, clinical features, panel-level ctDNA features, and target variables 
            # from the bootstrap training and validation dataloaders for the IMPower150 dataset, 
            # to use as input for retraining the output layer of the SHERLOC model on the bootstrap sample.
            for i, data in enumerate(subtrainloader, 0):
                    inputs, targets, other_inputs, agg_inputs, gene_encoded_inputs = data
            for l, vdata in enumerate(subvalloader, 0):
                    vinputs, vtargets, vother_inputs, vagg_inputs, vgene_encoded_inputs = vdata
                    val = tt.tuplefy([vinputs.float(), vagg_inputs.float(),vother_inputs.float(), vgene_encoded_inputs.float()], vtargets)

            # Retrain the output layer of the SHERLOC model on the bootstrap sample, and save the weights of 
            # the longitudinal representations of each feature.
            log = model_bt.fit([inputs.float(), agg_inputs.float(),other_inputs.float(), gene_encoded_inputs.float()], targets, batch_size, epochs, callbacks, verbose,val_data=val.repeat(10).cat())        
            weight_time_models.append(model_bt.net.filter1.weight.detach().cpu().numpy().flatten())
            weights_models.append(model_bt.net.fc_1.weight.detach().cpu().numpy().flatten())
            
    # Create DataFrames for the weights of the longitudinal representations of each feature across all bootstrap samples, 
    # and calculate confidence intervals and p-values using the correction of Nadeau and Bengio.    
    df_bt_weight_time = pd.DataFrame(weight_time_models)
    df_bt_weights = pd.DataFrame(weights_models)
    resultats_ic = calculate_corrected_confidence_intervals(
        df_bt_weights, 
        n_train=df_sub_train.shape[0], 
        n_test=df_sub_val.shape[0]
    )
    
    # Define the names of the features corresponding to the weights in the SHERLOC model, including the gene-level features,
    # the panel-level aggregated ctDNA features, and the clinical features, to use as index for the results DataFrame with 
    # confidence intervals and p-values.
    list_of_agg = ['Max VAF [1]','Max VAF [2]','Mean VAF [1]', 'Mean VAF [2]','cfDNA concentration [1]', 'cfDNA concentration [2]','Number driver mut. [1]','Number driver mut. [2]']
    cln_features = ['PDL1 High','Nb metastases','Baseline ECOG','Age', 'Sex Female','Sum Lesion Diameters Baseline',
                    'Treatment ABCP','Treatment BCP']
    pretrained_SB_features = ['BL Representation 1', 'C2D1 Representation 1',
        'C3D1 Representation 1', 'C4D1 Representation 1','BL Representation 2', 'C2D1 Representation 2',
        'C3D1 Representation 2', 'C4D1 Representation 2']
    names = gene_list_restricted+list_of_agg+cln_features+pretrained_SB_features
    resultats_ic.index = names

    # Create a DataFrame to store the weights of the longitudinal representations of each feature from the original SHERLOC model, 
    # and add these weights to the results DataFrame with confidence intervals and p-values for comparison.
    df_genes_time_weights = pd.DataFrame(model.net.filter1.weight.detach().cpu().numpy())
    df_genes_time_weights.columns = timepoints
    for col in timepoints:
        resultats_ic.loc[gene_list_restricted,col] = df_genes_time_weights[col][0]
    df_plot_agg = pd.DataFrame() 
    for j in range(len(model.net.mlps)):
        df_temp = pd.DataFrame(model.net.mlps[j][0].weight.detach().cpu().numpy())
        df_temp.columns = timepoints
        df_plot_agg = pd.concat([df_plot_agg, df_temp])
    df_plot_agg.index = list_of_agg
    for col in timepoints:
        for agg_feature in list_of_agg:
            resultats_ic.loc[agg_feature,col] = df_plot_agg.loc[agg_feature,col]
    resultats_ic = resultats_ic.fillna(0)
    df = resultats_ic.reset_index(drop=False, inplace=False)
    df.to_csv(path_save_results_file, index=True)

# Run the explainability function with the specified datasets and parameters to obtain 
# the weights, confidence intervals, and p-values for the longitudinal representations 
# of each feature in the SHERLOC model trained on the IMpower150 dataset.

timepoints = ['BL','C2D1','C3D1','C4D1']
d_channel = len(timepoints)
df_pretraining = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_mutations.txt', sep='\t')
df_genes = pd.read_csv(''.join(("/Users/aaronmamann/Documents/ImPower150_Genes_",str(d_channel),".csv")),index_col=0)
df_cln_vars = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_clinical_sample.txt', sep='\t', skiprows=4)
df_clinical_MSK = pd.read_csv('/Users/aaronmamann/Downloads/msk_chord_2024/data_clinical_patient.txt', sep='\t', skiprows=4)
df_list_genes = pd.read_csv(''.join(("/Users/aaronmamann/Documents/ImPower150_geneskept_restricted_",str(d_channel),".csv")),index_col=0)
df_gene_ctDNA = pd.read_csv("/Users/aaronmamann/Documents/ImPower150_OS_Interpretation_4.csv",index_col=0) 
df_panel_ctDNA = pd.read_csv("/Users/aaronmamann/Documents/ImPower150_AggFeatures.csv",index_col=0)
path_save_results_file = "/Users/aaronmamann/Documents/SHERLOC_Results.csv"

explainability(timepoints=timepoints, df_pretraining=df_pretraining, df_genes=df_genes, 
               df_cln_vars=df_cln_vars, df_clinical_MSK=df_clinical_MSK, 
               df_list_genes=df_list_genes, df_gene_ctDNA=df_gene_ctDNA, 
               df_panel_ctDNA=df_panel_ctDNA, path_save_results_file=path_save_results_file)
