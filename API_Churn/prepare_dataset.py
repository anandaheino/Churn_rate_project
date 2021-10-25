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

    
    def _load_dataframe(self):
        """Loads a DataFrame opening from the path passed when instantiating the class."""
        df = pd.read_csv(self.path, sep=';', index_col=False, low_memory=False)
        self.df = df
        return self.df

    def _split_df_customer_type(self):
        """Receives the initial DataFrame `df`, splits the clients according to:
        1. TYPE_BILLING: Commodities customers only
        2. Positive GROSS MARGIN (disregarding null and negative purchases)
        3. TYPE_HARVEST defines the purchase profiles, being customers who buy:
                - Only in winter
                - Only in summer
                - In all vintages (Full-year)
        Returns:
            customers_fullyear, customers_winter, customers_summer: lists containing the customers of each purchase profile.
            """
        df = self.df
        # %% Split DF by client type, filtering by COMMODITIES and TYPE_HARVEST
        customers_summer = list(
            df[(df['TYPE_HARVEST'] == 'SUMMER') & (df['GROSS_MARGIN'] > 0) & (df['TYPE_BILLING'] == 'COMMODITIES')][
                'COD_CLIENTE'].unique())
        customers_winter = list(
            df[(df['TYPE_HARVEST'] == 'INVERNO') & (df['GROSS_MARGIN'] > 0) & (df['TYPE_BILLING'] == 'COMMODITIES')][
                'COD_CLIENTE'].unique())
        customers_fullyear = []
        for v in customers_summer:
            for i in customers_winter:
                if i == v:
                    customers_fullyear.append(i)
        ### Removing fullyear customers from winter and summer
        for n in customers_fullyear:
            for v in customers_summer:
                if v == n:
                    customers_summer.remove(v)
            for i in customers_winter:
                if i == n:
                    customers_winter.remove(i)
        self.customers_fullyear, self.customers_winter, self.customers_summer = customers_fullyear, customers_winter, customers_summer
        return self.customers_fullyear, self.customers_winter, self.customers_summer



