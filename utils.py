import numpy as np
import pandas as pd
from scipy import stats

def calculate_corrected_confidence_intervals(df_coefs, n_train, n_test):
    
    """
    Compute the 95% confidence intervals for model coefficients using the Nadeau and 
    Bengio correction for repeated cross-validation.
    
    Parameters:
    df_coefs (pd.DataFrame): DataFrame containing the coefficients from each iteration of 
        the repeated cross-validation. Each row corresponds to an iteration (100 iterations) 
        and each column corresponds to a coefficient of the model. The DataFrame should 
        have dimensions (100, number_of_coefficients).
    n_train (float): Proportion of the dataset used for training in each iteration.
    n_test (float): Proportion of the dataset used for testing in each iteration.
    
    Returns:
    pd.DataFrame: Résumé avec Moyenne, Borne Inférieure et Borne Supérieure.
    """
    
    # Number of iterations for repeated cross-validation
    n = len(df_coefs)  
    
    # Compute the mean and variance of the coefficients across iterations
    means = df_coefs.mean()
    variances = df_coefs.var(ddof=1)
    
    # Apply the Nadeau and Bengio correction to the variance estimates to account 
    # for the dependence between iterations in repeated cross-validation.
    # Formula : Corrected_Variance = Variance * (1/n + n_test/n_train)
    correction_factor = (1/n) + (n_test/n_train)
    corrected_variances = variances * correction_factor
    corrected_std_errors = np.sqrt(corrected_variances)
    
    # Compute the 95% confidence intervals using the corrected standard errors and 
    # the t-distribution critical value for n-1 degrees of freedom.
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin_of_error = t_critical * corrected_std_errors
    lower_bound = means - margin_of_error
    upper_bound = means + margin_of_error
    
    # Calculate t-statistics and p-values for testing the null hypothesis that each coefficient is equal to zero.
    t_stats = means / corrected_std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 1))
    
    # Create a DataFrame to summarize the results, including the mean coefficient, confidence intervals, corrected 
    # standard error, t-statistic, and p-value for each coefficient.
    results = pd.DataFrame({
        'Moyenne': means,
        'IC_Inf_95': lower_bound,
        'IC_Sup_95': upper_bound,
        'Ecart_Type_Corrige': corrected_std_errors,
        't_stat': t_stats,
        'p_value': p_values
    })
    
    return results
