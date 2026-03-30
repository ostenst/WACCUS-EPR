"""
Created on May 26, 2015

@author: jhkwakkel
"""
import pandas as pd
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)

def filter_by_decision(experiments, outcomes, decision_values):
    """
    Filters the experiments and outcomes based on the categorical value of 'decision' and prints row counts.
    """

    initial_rows = len(experiments)

    # Ensure decision_values is a list
    if not isinstance(decision_values, list):
        decision_values = [decision_values]

    # Generate the correct column names (e.g., "decision_ref", "decision_amine", ...)
    decision_columns = [f"decision_{val}" for val in decision_values]

    # Create a filter mask: Select rows where at least one of the decision columns is 1
    mask = experiments[decision_columns].sum(axis=1) > 0

    # Apply filtering
    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes[mask]  # Keep outcomes aligned

    filtered_rows = len(filtered_experiments)

    print(f"Initial number of rows: {initial_rows}")
    print(f"Number of rows after filtering: {filtered_rows}")

    return filtered_experiments, filtered_outcomes

def classify(data):
    # Get the n_plants outcome
    result = data["n_plants"]
    classes = result >= 8  # More than 8 plants are built (good scenario)
    return classes


def filter_by_box(results, df_boxes, box_name, **categorical_filters):
    """
    Filters the experiments and corresponding outcomes based on the numerical limits of a given box.
    
    Parameters:
    - results: tuple of (experiments, outcomes) dataframes
    - df_boxes: DataFrame containing box boundaries
    - box_name: Name of the box to filter by (e.g., "box 1")
    - categorical_filters: Manually specified categorical conditions 
      (e.g., decision=["oxy", "clc"], Auction=True)
    
    Returns:
    - (filtered_experiments, filtered_outcomes): Tuple of filtered DataFrames
    """
    experiments, outcomes = results  
    print("NOTE: the user must manually specify what categorical features to include")
    print(f"Original number of experiments: {len(experiments)}")

    min_cols = df_boxes.loc[:, (box_name, "min")]
    max_cols = df_boxes.loc[:, (box_name, "max")]

    mask = pd.Series(True, index=experiments.index)
    for col in min_cols.index:
        if col in experiments.columns and pd.api.types.is_numeric_dtype(experiments[col]):
            mask &= (experiments[col] >= min_cols[col]) & (experiments[col] <= max_cols[col])

    for cat_col, cat_value in categorical_filters.items():
        if cat_col in experiments.columns:
            if isinstance(cat_value, list):  # If multiple categories are specified
                mask &= experiments[cat_col].isin(cat_value)
            else:  # If it's a single value (string, boolean, etc.)
                mask &= experiments[cat_col] == cat_value

    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes.loc[mask] 

    print(f"Filtered number of experiments: {len(filtered_experiments)}")

    return filtered_experiments, filtered_outcomes

def filter_by_feature_limits(results, feature_limits):
    """
    Filters experiments and outcomes based on hardcoded feature limits.

    """
    experiments, outcomes = results
    initial_rows = len(experiments)

    mask = pd.Series(True, index=experiments.index)

    for feature, limit in feature_limits.items():
        if isinstance(limit, tuple):  # Numeric range (min, max)
            mask &= (experiments[feature] >= limit[0]) & (experiments[feature] <= limit[1])
        else:  # One-hot encoded categorical filter (1 or 0)
            mask &= (experiments[feature] == limit)

    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes[mask]  # Keep outcomes aligned

    filtered_rows = len(filtered_experiments)

    print(f"Initial number of rows: {initial_rows}")
    print(f"Number of rows after filtering: {filtered_rows}")

    return filtered_experiments, filtered_outcomes

def count_classifications(filtered_outcomes, classify):

    classifications = classify(filtered_outcomes)
    
    count_true = classifications.sum()  
    count_false = len(classifications) - count_true 

    print(f"Number of 'True' classifications: {count_true}")
    print(f"Number of 'False' classifications: {count_false}")

    return count_true, count_false

if __name__ == "__main__":

    experiments = pd.read_csv("results/experiments_ccu.csv")
    outcomes = pd.read_csv("results/outcomes_ccu.csv")

    # Convert boolean columns to 1/0
    bool_columns = experiments.select_dtypes(include=["bool"]).columns
    experiments[bool_columns] = experiments[bool_columns].astype(int)

    # One-hot encode categorical string columns (object dtype)
    categorical_columns = experiments.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        experiments = pd.get_dummies(experiments, columns=categorical_columns, drop_first=False)
        # Ensure all one-hot encoded columns are in integer format (1/0 instead of True/False)
        for col in categorical_columns:
            encoded_columns = [c for c in experiments.columns if c.startswith(f'{col}_')]
            experiments[encoded_columns] = experiments[encoded_columns].astype(int)

    # Show categorical encoding results
    categorical_cols = [col for col in experiments.columns if col.startswith(('storage_', 'decision_'))]
    if categorical_cols:
        print(f"Categorical columns encoded: {categorical_cols}")
        for col in categorical_cols:
            print(f"  {col}: {experiments[col].sum()} cases")
    
    print(f"\nData loaded: {experiments.shape[0]} experiments, {experiments.shape[1]} features")
    
    # Check classification distribution
    classifications = classify(outcomes)
    print(f"Classification: {classifications.sum()} True, {(~classifications).sum()} False")
    
    results = (experiments, outcomes)
    
    cart_alg = cart.setup_cart(results, classify, mass_min=0.05)
    cart_alg.build_tree()
    print("Tree built successfully")
    df = cart_alg.boxes_to_dataframe()
    print(f"\nBoxes dataframe shape: {df.shape}")
    print("Boxes found:")
    print(df.head())
    
    # Try to show tree without pydot (skip if it fails)
    try:
        cart_alg.show_tree()
    except ImportError as e:
        print(f"Visualization skipped (missing dependency): {e}")
    except Exception as e:
        print(f"Visualization failed: {e}")

    plt.show()