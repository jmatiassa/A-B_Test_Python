import pandas as pd
from scipy.stats import chi2_contingency

def chi_squared_test(df, group_col, target_col):
    """
    Perform a Chi-square test to compare the conversion rate between two groups.
    
    Parameters:
    - df: DataFrame with the data.
    - group_col: Name of the column containing the groups (e.g. ‘test group’).
    - target_col: Name of the column containing the conversion variable (e.g. ‘converted’).
    """
    
    # Create contingency table
    contingency_table = pd.crosstab(df[group_col], df[target_col])
    
    # Chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Results in a dictionary
    results = {
        'Contingency Table': contingency_table,
        'Chi2 Stat': chi2_stat,
        'P-value': p_value,
        'Degrees of Freedom': dof,
        'Expected Frequencies': expected,
        'Statistical Significance (0.05)': 'Significant with 0.05 significance' if p_value < 0.05 else 'Not Significant',
        'Statistical Significance (0.01)': 'Significant with 0.01 significance' if p_value < 0.01 else 'Not Significant'
    }
    
    # Show results
    for key, value in results.items():
        print(f"{key}: {value}\n")



