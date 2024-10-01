import pandas as pd
import os
import re
from gensim.models.callbacks import CallbackAny2Vec
import logging


def extract_id_from_uri(uri):
    """ Extract the ID from a URI string. """
    return uri.split(':')[-1]

def normalize_name(name):
    if pd.isna(name): return name
    name = name.lower()
    name = re.sub(r"[.,#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def reconstruct_uri(uri, type):
    if type == 'track':
        return "spotify:track:" + uri
    elif type == 'artist':
        return "spotify:artist:" + uri
    elif type == 'album':
        return "spotify:album:" + uri

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

class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0
        self.losses = {}
    
    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        loss_in_epoch = current_loss - self.loss_previous_step
        self.loss_previous_step = current_loss
        logging.info(f"Model {model} - Epoch {self.epoch} Loss: {loss_in_epoch}")
        self.losses[self.epoch] = loss_in_epoch
        self.epoch +=1
    
    def on_train_end(self, model):
        logging.info("Losses per epoch:")
        logging.info(self.losses)
        
