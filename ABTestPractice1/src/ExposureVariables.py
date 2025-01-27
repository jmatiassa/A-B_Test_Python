import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pointbiserialr
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm
import plotly.express as px

def point_biserial_correlation(df, col_x, col_y):
    """
    Calculates the point biserial correlation between two columns.
    
    Args:
    - df (DataFrame): The DataFrame containing the data.
    - col_x (str): Name of the continuous column (independent variable).
    - col_y (str): Name of the binary column (dependent variable).
    
    Returns:
    - correlation (float): The correlation coefficient.
    - p_value (float): The associated p-value.
    """
    # Check if the columns exist in the DataFrame
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError("One or both columns do not exist in the DataFrame.")
    
    # Compute the point biserial correlation
    correlation, p_value = pointbiserialr(df[col_x], df[col_y])
    
    # Print results
    print(f"Point Biserial Correlation ({col_x} vs {col_y}): {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"P-value:(full precision): {p_value:.20f}")

    
    if p_value < 0.05:
        print("The relationship is statistically significant.")
    else:
        print("There is no sufficient evidence to support a significant relationship.")
    
    return correlation, p_value




def logistic_regression_with_quadratic(df, xcol, ycol):
    # Step 1: Create a squared term for the independent variable (xcol)
    df[f'{xcol}_squared'] = df[xcol] ** 2
    
    # Step 2: Prepare the independent (X) and dependent (y) variables
    X = df[[xcol, f'{xcol}_squared']]  # Include the quadratic term
    y = df[ycol]
    
    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 4: Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Step 5: Get the model coefficients and intercept
    print(f"Model coefficients for {xcol} and {xcol}^2:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    
    # Step 6: Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Step 7: Evaluate the model
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Optional: Use statsmodels to get a more detailed statistical summary
    # Add a constant to the independent variables for the intercept term
    X_with_intercept = sm.add_constant(X)
    
    # Fit the model with statsmodels
    logit_model = sm.Logit(y, X_with_intercept)
    result = logit_model.fit()
    
    # Show the summary of the logistic regression model
    print("\nLogistic Regression Summary (statsmodels):")
    print(result.summary())



def plot_conversion_ratio_with_scroll_plotly(df, xcol, ycol):
    # Calculate conversion ratio
    conversion_ratio = df.groupby(xcol).apply(
        lambda x: x[ycol].sum() / len(x)
    ).reset_index(name='conversion_ratio')
    
    # Create an interactive plot with Plotly
    fig = px.line(conversion_ratio, x=xcol, y='conversion_ratio', 
                  title=f'Conversion Ratio by {xcol}', 
                  labels={xcol: f'{xcol} (Total Ads)', 'conversion_ratio': 'Conversion Ratio'})
    
    # Show the plot
    fig.update_traces(mode='markers+lines')
    fig.show()

def plot_conversion_by_intervals(df, xcol, ycol):
    # Calculate the quartiles (use pandas' quantile function)
    q1, q2, q3 = df[xcol].quantile([0.25, 0.5, 0.75])

    # Define the intervals based on the quartiles
    intervals = [
        (df[xcol] <= q1, f"0 - {q1}"),
        ((df[xcol] > q1) & (df[xcol] <= q2), f"{q1} - {q2}"),
        ((df[xcol] > q2) & (df[xcol] <= q3), f"{q2} - {q3}"),
        (df[xcol] > q3, f"{q3} - max({df[xcol].max()})")
    ]
    
    # Create a new column for the interval each row belongs to
    df['interval'] = None
    for condition, label in intervals:
        df.loc[condition, 'interval'] = label

    # Calculate conversion ratio for each interval
    conversion_by_interval = df.groupby('interval').apply(
        lambda x: x[ycol].sum() / len(x)  # Sum of conversions / Total count in each interval
    ).reset_index(name='conversion_ratio')

    # Plot the conversion ratio by interval using plotly for interactivity
    fig = px.bar(conversion_by_interval, x='interval', y='conversion_ratio', 
                 title=f'Conversion Ratio by Intervals of {xcol}', 
                 labels={'interval': 'Interval', 'conversion_ratio': 'Conversion Ratio'})

    # Show the plot
    fig.show()



def plot_conversion_rate_by_hour(df, colx, coly):
    """
    Plots the conversion rate by a categorical hour-based column (colx) 
    and adds a horizontal line for the average conversion rate.

    Parameters:
    - df: DataFrame
    - colx: Column name representing the hour-based feature (e.g., 'most ads hour')
    - coly: Column name representing the binary conversion feature (e.g., 'converted')
    """
    # Grouping by hour (colx) and calculating the mean conversion rate (coly)
    conversion_rate_by_hour = df.groupby(colx)[coly].mean()
    
    # Plotting the conversion rate by hour
    plt.figure(figsize=(10, 6))
    conversion_rate_by_hour.plot(kind='bar')

    # Adding a horizontal line for the average conversion rate
    average_conversion_rate = conversion_rate_by_hour.mean()
    plt.axhline(y=average_conversion_rate, color='r', linestyle='--', label=f'Average Conversion Rate per Hour: {average_conversion_rate:.4f}')

    # Adding labels and title
    plt.title(f'Conversion Rate by {colx}')
    plt.xlabel(f'{colx} of the Day')
    plt.ylabel('Conversion Rate')
    plt.xticks(rotation=45)

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()












