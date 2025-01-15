from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any


def correct_using_interpolation(df: pd.DataFrame, columns_to_check: List[str]) -> pd.DataFrame:
    """
    Corrects values in specified columns of a DataFrame using interpolation.
    This function takes a DataFrame and a list of column names to check. It iterates through the specified columns
    (excluding the first and last columns in the list) and corrects values that are greater than the previous column's
    value by interpolating between the previous and next column values.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be corrected.
    columns_to_check (List[str]): A list of column names to check and correct.
    Returns:
    pd.DataFrame: A new DataFrame with corrected values in the specified columns.
    """

    df_corrected = df.copy()

    for col_index in range(1, len(columns_to_check) - 1):
        current_col = columns_to_check[col_index]
        prev_col = columns_to_check[col_index - 1]
        next_col = columns_to_check[col_index + 1]

        condition = (df[current_col] > df[prev_col]) & pd.notnull(df[current_col]) & pd.notnull(df[prev_col])
        corrected_value = (df[prev_col] + df[next_col]) / 2
        df_corrected.loc[condition, current_col] = corrected_value.round()
    return df_corrected


def check_logical_consistency(df)-> List[Dict[str, Any]]:
    """
    Checks the logical consistency of a DataFrame by ensuring that the values in specified columns
    follow a logical progression. Specifically, it checks if the value in a column is greater than
    the value in the previous column, which would indicate an inconsistency.
    Parameters:
    df (pandas.DataFrame): The DataFrame to check for logical consistency.
    Returns:
    list: A list of dictionaries, each containing details of an inconsistent row. Each dictionary
        contains the following keys:
        - 'row_index': The index of the row with inconsistency.
        - 'prev_col': The name of the previous column.
        - 'current_col': The name of the current column.
        - 'next_col': The name of the next column.
        - 'current_value': The value in the current column.
        - 'next_value': The value in the next column.
    """
   
    df_copy = df.copy()

    columns_to_check = ['website_clicks', 'searches', 'view_content', 'add_to_cart', 'purchase']
    inconsistent_details = []
    for i, row in df_copy.iterrows():
        for col_index in range(1, len(columns_to_check) - 1):
            current_col = columns_to_check[col_index]
            next_col = columns_to_check[col_index + 1]
            prev_col = columns_to_check[col_index - 1]
            # Check if the value in the current column is greater than in the previous column
            if pd.notnull(row[current_col]) and pd.notnull(row[next_col]):
                if row[current_col] > row[prev_col]:
                    inconsistent_details.append({
                        'row_index': i,
                        'prev_col': prev_col,
                        'current_col': current_col,
                        'next_col': next_col,
                        'current_value': row[current_col],
                        'next_value': row[next_col]
                    })
    return inconsistent_details


def log_anomalies(details: List[Dict[str, Any]], name: str = '', when: str = '') -> None:
    """
    Logs details about anomalies in a dataset.
    Args:
        details (List[Dict[str, Any]]): A list of dictionaries containing details about each anomaly.
            Each dictionary should have a 'row_index' key indicating the row number of the anomaly.
        name (str, optional): The name of the dataset. Defaults to an empty string.
        when (str, optional): A string indicating when the anomalies were detected, either 'before' or 'after'.
            Defaults to an empty string. If not 'before' or 'after', it will be set to 'before/after'.
    Returns:
        None
    """
   
    length = len(details)
    if when not in ['before', 'after']:
        when = 'before/after'

    print(f"Anomalies in {name} dataset {when} correction:")
    print(f"Sum of anomalies: {length}")

    if length != 0:
        print("Anomalies in rows:")
        for detail in details:
            print(f"Row {detail['row_index']}")


def clean_column_names(df: pd.DataFrame, columns_to_drop: List[str] = ['Campaign Name', 'Date']) -> pd.DataFrame:
    """
    Cleans the column names of a DataFrame by stripping whitespace, replacing spaces with underscores,
    converting to lowercase, and removing specific substrings. Optionally drops specified columns.
    Args:
        df (pd.DataFrame): The input DataFrame whose column names need to be cleaned.
        columns_to_drop (List[str], optional): A list of column names to drop from the DataFrame. 
                                               Defaults to ['Campaign Name', 'Date'].
    Returns:
        pd.DataFrame: A new DataFrame with cleaned column names and specified columns dropped.
    """
    
    clean_df = df.copy()
    print(clean_df.columns)
    clean_df.columns = clean_df.columns.str.strip().str.replace('# of ','').str.replace(' [USD]','').str.replace(' ', '_').str.lower()
    clean_df = clean_df.drop(columns=columns_to_drop, errors='ignore')
    return clean_df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns in a DataFrame to the smallest possible integer subtype.
    This function creates a copy of the input DataFrame and iterates over all columns
    with numeric data types. If a column does not contain any null values, it is 
    downcast to the smallest possible integer subtype to save memory.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing numeric columns to be downcast.
    Returns:
    pd.DataFrame: A new DataFrame with numeric columns downcast to the smallest possible integer subtype.
    """
    
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['number']).columns:
        if not df_copy[col].isnull().any():
            df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
    return df_copy


def plot_comparison_column(df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> None:
    """
    Plots a bar chart comparing the total values of a specified column between two DataFrames.
    Parameters:
    df (pd.DataFrame): The control DataFrame.
    test_df (pd.DataFrame): The test DataFrame.
    column (str): The column name to compare.
    Returns:
    None
    """
    
    total_spend = {
        'Control': df[column].sum(),
        'Test': test_df[column].sum()
    }
    plt.bar(total_spend.keys(), total_spend.values(), color=['lightblue', 'lightgreen'])
    plt.title(f'Total {column} Comparison')
    plt.ylabel(f'Total {column}')
    plt.xlabel('Groups')
    plt.show()


def calculate_conversion_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the conversion rate from a DataFrame.
    This function takes a DataFrame containing 'purchase' and 'impressions' columns,
    calculates the conversion rate as (purchase / impressions) * 100, and returns
    a new DataFrame with an additional 'Conversion Rate' column.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'purchase' and 'impressions' columns.
    Returns:
    pd.DataFrame: A new DataFrame with an additional 'Conversion Rate' column.
    Raises:
    ValueError: If the input DataFrame does not contain the required columns.
    """
    
    required_columns = ['purchase', 'impressions']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame")

    df_copy = df.copy()
    df_copy['Conversion Rate'] = (df_copy['purchase'] / df_copy['impressions']) * 100
    return df_copy


def fill_missing_with_rounded_mean(df) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame with the rounded mean of each numeric column.
    Parameters:
    df (pandas.DataFrame): The input DataFrame with potential missing values.
    Returns:
    pandas.DataFrame: A new DataFrame with missing values filled with the rounded mean of each numeric column.
    """
    # Copy of DataFrame to avoid changing the original
    df_copy = df.copy()
    
    # Processing numeric columns
    for col in df_copy.select_dtypes(include=['float', 'int']).columns:
        mean_value = df_copy[col].mean()  # Calculate mean
        df_copy[col] = df_copy[col].fillna(round(mean_value))  # Fill NaN with rounded mean
    
    return df_copy
