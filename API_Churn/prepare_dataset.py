import pandas as pd
from functools import reduce
import time
pd.options.mode.chained_assignment = None  # remove warnings from pandas: default='warn'


class PrepareDataset:
    """Receives the path of the .csv or .txt file of the dataset to be processed and the type of treatment.
    Args:
        - path: (str) .csv or .txt file directory path.
        - dataset_predict: (bool) defines whether the treatment will be for the classification model prediction data (True)
                            or if it will be for the model training data (False: default).
        - model_type: (str) if it is "churn" the data will be treated for input into the Churn prediction model
                     but if it is "recur", the data treatment will be to train the recurrence model.
    Returns:
        - df_input_model: pd.DataFrame containing the columns for input in the classification model.
        - df_predict_model: pd.DataFrame containing the X columns for prediction using the ranking model."""

def __init__(self, path: str, dataset_predict = False, model_type: str = 'churn'):
        self.path = path
        self.dataset_predict = dataset_predict
        self.model_type = model_type

        