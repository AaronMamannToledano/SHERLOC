library(nlme)
library(data.table)
library(dplyr)
library(rslurm)
library(writexl)
library(survival)
library(JMbayes2)
library(GLMMadaptive)


Joint_Model <- function(df, df_indices, experiment_number, d_channel, seed, threshold_feature_frequency=0.05){  


" 

  This function implements a joint model for longitudinal and survival data, 
  using the JMbayes2 package in R. The function takes as input a dataframe with longitudinal 
  and survival data, a dataframe with indices for train/test splits, the experiment number to run, 
  the number of channels (timepoints) in the longitudinal data, and a threshold for feature frequency 
  to include in the model.

  df: dataframe with longitudinal and survival data for the IMPower150 dataset, 
  including columns for time to event, event indicator, timepoint, and longitudinal features 
  (e.g. gene-level ctDNA VAFs, panel-level ctDNA features) and clinical variables 
  df_indices: dataframe with indices for train/test splits for the IMPower150 dataset
  experiment_number: integer from 1 to 100 indicating which train/test split to use for this run of the function
  d_channel: integer from 2 to 4 indicating the number of channels (timepoints) in the longitudinal data
  threshold_feature_frequency: minimum frequency of a feature in the training set to be included in the model, 
  default is 0.05 (i.e. feature must be present in at least 5% of patients in the training set)

"

# Define timepoints for evaluation of model performance with time-dependent AUC and Brier score metrics
times = seq(2.5,40,5)

# Convert timepoint variable to numeric values for modeling
df[which(df$timepoint=='BL'),'timepoint']=0
df[which(df$timepoint=='C2D1'),'timepoint']=1*(3/4)
df[which(df$timepoint=='C3D1'),'timepoint']=2*(3/4)
df[which(df$timepoint=='C4D1'),'timepoint']=3*(3/4)
df$timepoint = as.numeric(df$timepoint)

# Select latent features from MSK-CHORD dataset and clinical features to include in the model 
genomic_interaction_last_index = (2*d_channel)-1
formula_surv = as.formula(paste0('Surv(TTEOS_rebaseline_BL,OS_event) ~ ',paste(unlist(list(c(paste0('X',0:genomic_interaction_last_index),colnames(df)[grep('cln_', colnames(df))]))), collapse = ' + ')))

# Create train/test splits based on indices provided in df_indices for the specified experiment number
# So that we used the same splits across different models for fair comparison of performance metrics
test_indices <- as.numeric(df_indices[experiment_number,])
train_indices <- setdiff(unique(df$index), test_indices)
df_train = df %>% filter(index %in% train_indices)
df_test = df %>% filter(index %in% test_indices)

# Select features to include in the model based on frequency in the training set 
# (default threshold is 5% frequency, i.e. feature must be present in at least 5% of patients 
# in the training set to be included in the model), 
# necessary to avoid convergence issues in the joint model fitting when including very rare features

features_list = list()
for (i in colnames(df %>% select(-c(TTEOS_rebaseline_BL, OS_event, index, timepoint, colnames(df)[grep('cln',colnames(df))], colnames(df)[grep('X',colnames(df))] )))){
    freq_var = length(unique(df_train[which(df_train[i]>0),'index']))/length(unique(df_train[['index']]))
    if (freq_var>threshold_feature_frequency){
        features_list = append(features_list, i)
    }
}

# Scale features in the training and test sets based on the mean and standard deviation of the training set,
# which is important for convergence of the joint model fitting and also ensures that feature values are on
# a comparable scale 
df_train[unlist(features_list)] <- scale(df_train[unlist(features_list)])
df_test[unlist(features_list)]  <- scale(df_test[unlist(features_list)])

# Set a random seed for reproducibility of results across runs of the function, since 
# the joint model fitting involves Markov Chain Monte Carlo sampling which can yield slightly
# different results across runs if the seed is not set.
set.seed(seed)

# Fit the joint model using the Cox proportional hazards model for the survival submodel 
# and linear mixed effects models for the longitudinal submodels, with the specified features 
# and time variable, and using 4000 iterations (increasing the default setting which was 
# 3500 iterations in the original function) and 1000 burn-in iterations (increasing the default setting 
# of 500 iterations) for the MCMC sampling in the joint model fitting.
CoxFit <- coxph(formula_surv,data=df_train %>% filter(timepoint == 0))
lme_list <- lapply(features_list, function(feat) {
  lme(as.formula(paste0(feat, " ~ timepoint")),random = ~ 1 | index,data=df_train)
})
names(lme_list) <- features_list
jointFit <- jm(CoxFit,lme_list,time_var = "timepoint", n_iter = 4000, n_burnin = 1000)

# Evaluate model performance on the test set using time-dependent Brier score and AUC metrics,
# which are commonly used metrics for evaluating survival prediction models, and store results
# for this experiment number to be returned by the function. The Brier score is integrated over time 
# from 2.5 to 40 months, and the AUC is computed at each timepoint in the times vector and averaged across timepoints 
# to yield a single integrated AUC value for this experiment number.
ibs = tvBrier(jointFit, newdata = df_test %>% filter(timepoint == 0), Tstart = 2.5, integrated=TRUE, Thoriz=40, cores = 1L, type_weights = c("IPCW"))[1]$Brier
auc = c()
i=1
for (time in length(seq(times))){
  auc[i] = tvAUC(jointFit, newdata = df_test %>% filter(timepoint == 0), Tstart = 2.1, Thoriz=time, cores = 1L, type_weights = c("IPCW"))[1]$auc
  i=i+1
}
auc_mean = mean(auc)

return(list(ibs, auc_mean))
}

# Read in the IMPower150 dataset for the specified number of channels (timepoints) in the longitudinal data,
# which will be used as input to the Joint_Model function. Execute the function for each of the 100 train/test splits defined in df_indices, 
# and save results to a file.
d_channel=4
df = read.csv(paste0("ImPower150_OS_Data_for_JointModels_R",as.character(d_channel),".csv"), row.names=1) 
df_indices = read.csv(paste0("ImPower150_OS_Test_Indices_for_R",as.character(d_channel),".csv"), row.names=1)
functions_to_export <- c('Joint_Model','df','df_indices','d_channel')
sopt <- list(time = "04:00:00", "ntasks" = 1, "cpus-per-task" = 1, partition = "cpu_med")

list_results_model <- slurm_apply(function(experiment_number) {
    Joint_Model(df=df, df_indices=df_indices, experiment_number=experiment_number, d_channel=d_channel, seed=3, threshold_feature_frequency=0.05)},
    params=data.frame(experiment_number=seq(1,100)),
    nodes=length(seq(1,100)),
    cpus_per_node=1,
    global_objects=functions_to_export,
    slurm_options=sopt)

res <- get_slurm_out(list_results_model, outtype = "raw")
df_results <- do.call(rbind, lapply(res, function(x) {data.frame(IBS = x[[1]],AUC = x[[2]])}))
filepath_status <- paste0("IBSandAUC_",as.character(d_channel),".csv")
write.csv(df_results, file=filepath_status, row.names = TRUE) 

