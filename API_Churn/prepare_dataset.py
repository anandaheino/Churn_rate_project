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
                'COD_CUSTOMER'].unique())
        customers_winter = list(
            df[(df['TYPE_HARVEST'] == 'WINTER') & (df['GROSS_MARGIN'] > 0) & (df['TYPE_BILLING'] == 'COMMODITIES')][
                'COD_CUSTOMER'].unique())
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


def _create_df_commodities(self):
        """Receives df and removes irrelevant columns, null data and filters the data according to TYPE_BILLING 
        equals to commodities, and only positive GROSS_MARGIN values.
        Returns:
            df_commodities: DataFrame containing the purchase data of input customers with positive gross margin."""

        df = self.df
        df = df.fillna(value=0)
        df.dropna(axis=0, inplace=True)
       
        # Filtering data by billing type and positive gross margin
        df_commodities = df[(df['TYPE_BILLING'] == 'COMODITIES') & (df['GROSS_MARGIN'] > 0)]
        df_commodities = df_commodities.drop(['TYPE_BILLING'], axis=1)
        # Reorganizando as colunas
        self.df_commodities = df_commodities[['COD_CUSTOMER', 'DATE_MOV', 'COD_HARVEST', 'TYPE_HARVEST', 'MIX_GROUP', 'TYPE_CONTRACT',
                                      'TYPE_MOV', 'REGION', 'QUANTITY', 'VALUE',
                                      'TRADE_DESCRIPTION', 'EVENT_DESCRIPTION', 'GROSS_MARGIN']]
        return self.df_commodities


    def _get_dummies_commodities(self):
        """Receives df_commodities and creates a column for each categorical data (dummies) with a numeric data that represents
        the data (as string) on a given line.
        Returns:
             df_ins_dumm: concatenation of all previously generated dummies
             dummies_cod_harvest: dummies referring to harvest codes"""
        df_commodities = self.df_commodities
        self.dummies_mix = pd.get_dummies(df_commodities['MIX_GROUP']).reset_index()
        self.dummies_reg = pd.get_dummies(df_commodities['REGION']).reset_index()
        self.dummies_type_movto = pd.get_dummies(df_commodities['TYPE_MOV']).reset_index()
        self.dummies_type_cont = pd.get_dummies(df_commodities['TYPE_CONTRACT']).reset_index()
        self.dummies_type_harvest = pd.get_dummies(df_commodities['TYPE_HARVEST']).reset_index()
        self.dummies_cod_harvest = pd.get_dummies(df_commodities['COD_HARVEST']).reset_index()
        self.dummies_even_com = pd.get_dummies(df_commodities['EVENT_DESCRIPTION']).reset_index()
        df_commodities = df_commodities.reset_index()

        dumies_t = [df_commodities, self.dummies_mix, self.dummies_cod_harvest, self.dummies_type_harvest,
                    self.dummies_reg,self.dummies_type_cont, self.dummies_type_movto, self.dummies_even_com]

        self.df_ins_dumm = reduce(lambda left, right: pd.merge(left,
                                                               right,
                                                               on='index'), dumies_t)
        try:
            self.df_ins_dumm.drop(['0_x', '0_y', 0], axis=1, inplace=True)
        
        except:
            self.df_ins_dumm.drop(['0_x','0.0_y'], axis=1, inplace=True)

        return self.df_ins_dumm, self.dummies_cod_harvest


    def _create_df_stats(self):
        """Create the DataFrame with statistical data from df_ins_dumm and dummies_cod_harvest. The value of
        GROSS MARGIN is applied to the harvest columns, witch were grouped ​with all purchases within that period.
        QUANTITY is applied to new columns created for each product of MIX_GROUP.
        Data are grouped and summed by COD_CUSTOMER.
        Returns:
            df_stat_commodities: DataFrame with informative statistical data on purchases by season,
            product, type of contract and event of each customer. """
        # apllying 'GROSS_MARGIN' to the 'COD_HARVEST' dummies
        df_value_harvest = self.df_ins_dumm[self.dummies_cod_harvest.columns.values].apply(
                                                                        lambda x: x * self.df_ins_dumm['GROSS_MARGIN'])
        self.df_ins_dumm[self.dummies_cod_harvest.columns.values] = df_value_harvest

        # applying column 'QUANTITY' to the 'MIX_GROUP' dummies
        df_value_mix = self.df_ins_dumm[self.dummies_mix.columns.values].apply(
            lambda x: x * self.df_ins_dumm['QUANTITY'])
        self.df_ins_dumm[self.dummies_mix.columns.values] = df_value_mix

        # Grouping by 'COD_CUSTOMER'
        df_ins_sum = self.df_ins_dumm.groupby('COD_CUSTOMER').sum()
        df_ins_sum.drop('index', axis=1, inplace=True)
        self.df_stat_commodities = df_ins_sum.round(2).copy()
        return self.df_stat_commodities


    def filter_df_per_customer(self, clients_list: list(), df_est: pd.DataFrame):
        """Receives a DataFrame containing the column COD_CUSTOMER and filters
            according to the values ​​within clients_list.
        This is a helper method of other methods that are called when instantiating the class."""

        filtered = df_est['COD_CUSTOMER'].isin(clients_list)
        df_customer = df_est[filtered]
        return df_customer


    def convert_harvest_to_date(self, df_complete: pd.DataFrame):
            """Receives a DataFrame containing the column COD_HARVEST and adds a DATE_HARVEST column according to harvest.
            This is a helper method of other methods that are called when instantiating the class."""

            # unique harvest codes
            Codigo_harvest = df_complete['COD_HARVEST'].unique()
            Codigo_harvest = list(Codigo_harvest)  
            Codigo_harvest = sorted(Codigo_harvest)  
            date_harvest_dict = {i: i.replace('SF-', '').split('/') for i in Codigo_harvest}
   
            date_harvest = {}
            for k, j in (date_harvest_dict.items()):
                if j[0] == j[1]:  # winter harvest 
                    date_harvest[k] = (f'20{j[1]}-07-01')

                elif j[0] != j[1]:  # summer harvest
                    date_harvest[k] = (f'20{j[1]}-01-01')

            # adds DATE_HARVEST using the date_harvest_dict with map function
            df_complete['DATE_HARVEST'] = (df_complete['COD_HARVEST'].map(date_harvest))

            return df_complete