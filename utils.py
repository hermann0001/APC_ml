import pandas as pd
import os

def extract_id_from_uri(uri):
    """ Extract the ID from a URI string. """
    return uri.split(':')[-1]

def check_csv(file_path, expected_rows, id_column=None):
    """
    Check if a CSV file exists, is non-empty, and optionally validate the presence of an ID column.

    Args:
    - file_path (str): The path to the CSV file.
    - expected_rows (int): List of expected rows in the CSV file.
    - id_column (str, optional): The name of the ID column to check for unique values.

    Returns:
    - bool: True if all checks pass, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exists.")
        return False
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    if df.empty:
        print(f"{file_path} is empty.")
        return False

    if len(df) != expected_rows:
        print(f"Some rows missing from {file_path}, actual length: {len(df)}")
        return False
    
    if id_column and id_column in df.columns:
        if df[id_column].isnull().any():
            print(f"{file_path} contains null values in the {id_column} column.")
            return False
        if not df[id_column].is_unique:
            print(f"{file_path} contains duplicate values in the {id_column} column.")
            return False
        
    print(f"{file_path} is correctly written.")
    return True

