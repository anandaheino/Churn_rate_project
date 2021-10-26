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

    def _split_df_stat_by_customer(self):
        """Uses the columns of the df_stat_commodities crops according to the current year.
        The harevsts are updated every six months, that is, the model will always receive updated data.
        Returns:
            Summer, winter and full-year DataFrames according to separate lists for each customer profile."""
        customer_winter, customer_summer, customer_fullyear = self.customer_winter, self.customer_summer, self.customer_fullyear

        current_year = time.strftime('%Y')[-2:]  # year: e.g. 2021 uses only two last numbers "21"
        current_month = time.strftime('%m')
           # between april and september
        if 4 <= int(current_month) and int(current_month) < 10:
            df_harvest = self.df_stat_commodities[[f'SF-{int(current_year)-6}/{int(current_year)-5}',f'SF-{int(current_year)-5}/{int(current_year)-5}',
                                            f'SF-{int(current_year)-5}/{int(current_year)-4}',  
                                            f'SF-{int(current_year)-4}/{int(current_year)-4}', f'SF-{int(current_year)-4}/{int(current_year)-3}',
                                            f'SF-{int(current_year)-3}/{int(current_year)-3}', f'SF-{int(current_year)-3}/{int(current_year)-2}',
                                            f'SF-{int(current_year)-2}/{int(current_year)-2}', f'SF-{int(current_year)-2}/{int(current_year)-1}',
                                            f'SF-{int(current_year)-1}/{int(current_year)-1}', f'SF-{int(current_year)-1}/{int(current_year)}',
                                            f'SF-{int(current_year)}/{int(current_year)}']]
                                                                     # last one is the winter harvest from the current year
          # between october and march
        if int(current_month) >= 10:
            df_harvest = self.df_stat_commodities[[f'SF-{int(current_year)-5}/{int(current_year)-5}', f'SF-{int(current_year)-5}/{int(current_year)-4}',  # cod_CLIENTE já é o índice
                                            f'SF-{int(current_year)-4}/{int(current_year)-4}', f'SF-{int(current_year)-4}/{int(current_year)-3}',
                                            f'SF-{int(current_year)-3}/{int(current_year)-3}', f'SF-{int(current_year)-3}/{int(current_year)-2}',
                                            f'SF-{int(current_year)-2}/{int(current_year)-2}', f'SF-{int(current_year)-2}/{int(current_year)-1}',
                                            f'SF-{int(current_year)-1}/{int(current_year)-1}', f'SF-{int(current_year)-1}/{int(current_year)}',
                                            f'SF-{int(current_year)}/{int(current_year)}', f'SF-{int(current_year)}/{int(current_year)+1}']]
                                                                     # last one is the summer harvest of next year
        if int(current_month) < 4:  
            df_harvest = self.df_stat_commodities[[f'SF-{int(current_year)-6}/{int(current_year)-6}', f'SF-{int(current_year)-6}/{int(current_year)-5}',  # cod_CLIENTE já é o índice
                                            f'SF-{int(current_year)-5}/{int(current_year)-5}', f'SF-{int(current_year)-5}/{int(current_year)-4}',
                                            f'SF-{int(current_year)-4}/{int(current_year)-4}', f'SF-{int(current_year)-4}/{int(current_year)-3}',
                                            f'SF-{int(current_year)-3}/{int(current_year)-3}', f'SF-{int(current_year)-3}/{int(current_year)-2}',
                                            f'SF-{int(current_year)-2}/{int(current_year)-2}', f'SF-{int(current_year)-2}/{int(current_year)-1}',
                                            f'SF-{int(current_year)-1}/{int(current_year)-1}', f'SF-{int(current_year)-1}/{int(current_year)}']]
                                                                     # last one is the summer harvest of the current year
        # Winter dataframe
        df_harvest.reset_index(inplace=True)
        df_winter = self.filter_df_per_customer(customer_winter, df_harvest)
        df_winter = df_winter.set_index('COD_CLIENTE').T.reset_index().rename(columns={'index': 'COD_HARVEST'})
        df_winter = self.convert_harvest_to_date(df_winter)
        df_winter['DATE_HARVEST'] = pd.to_datetime(df_winter['DATE_HARVEST'])
        df_winter = df_winter.set_index('DATE_HARVEST').drop('COD_HARVEST', axis=1)
        drop_winter = df_winter[df_winter.index.month == 1]
        self.df_winter = df_winter.drop(drop_winter.index)

        # summer dataframe
        df_summer = self.filter_df_per_customer(customer_summer, df_harvest)
        df_summer = df_summer.set_index('COD_CLIENTE').T.reset_index().rename(columns={'index': 'COD_HARVEST'})
        df_summer = self.convert_harvest_to_date(df_summer)
        df_summer['DATE_HARVEST'] = pd.to_datetime(df_summer['DATE_HARVEST'])
        df_summer = df_summer.set_index('DATE_HARVEST').drop('COD_HARVEST', axis=1)
        drop_summer = df_summer[df_summer.index.month == 7]
        self.df_summer = df_summer.drop(drop_summer.index)

        # fullyear dataframe
        df_fullyear = self.filter_df_per_customer(customer_fullyear, df_harvest)
        df_fullyear = df_fullyear.set_index('COD_CLIENTE').T.reset_index().rename(columns={'index': 'COD_HARVEST'})
        df_fullyear = self.convert_harvest_to_date(df_fullyear)
        df_fullyear['DATE_HARVEST'] = pd.to_datetime(df_fullyear['DATE_HARVEST'])
        self.df_fullyear = df_fullyear.set_index('DATE_HARVEST').drop('COD_HARVEST', axis=1)

        return self.df_winter, self.df_summer, self.df_fullyear


    def add_churn_recur_values(self, df_full):
            """Receives a DataFrame with customer purchases grouped by harvest period and
            checks if the customer stayed at least two harvest without buying with the company.
            It will be considered churn when, after the first purchase, there is more than 2 
            zeros in a row. On the other hand, if the customer with positive churn starts to buy
            again at the company, than we have positive recurrence.
            This is a helper method of other methods that are called when instantiating the class."""

            df_full['churn'] = 0
            for i in range(len(df_full)):
                for c in range(len(df_full.values[0]) - 1):  # without the last column: churn
                    if df_full.values[i][c] != 0:             # does not count until the client starts to buy
                        for buy in range(c, c + len(df_full.values[i][c:-1]) - 1):  # first to last purchase
                            if (df_full.values[i][buy] == 0 and df_full.values[i][buy + 1] == 0):
                                df_full.iloc[i, -1] = 1
                                break  # when the algorithm finds 2 following zeros
                        break

            df_full['recur'] = 0
            for i in range(len(df_full)):
                if (df_full['churn'][i] == 1):
                    for c in range(len(df_full.values[0]) - 2): 
                        if df_full.values[i][c] != 0:
                            for buy in range(c, c + len(df_full.values[i][c:-2]) - 2):  
                                if (df_full.values[i][buy] == 0 and df_full.values[i][buy + 1] == 0 and df_full.values[i][buy + 2] != 0):
                                    df_full.iloc[i, -1] = 1
                                    break
                            break
            return df_full


    def count_time_off(self, df_full: pd.DataFrame):
        """Receives a DataFrame classified with churn and recurrence to check the time off. 
        If recurrence is equal to one (positive), this method counts the total harvests 
        without purchases between harvests with purchases, including the 2 off from churn.
        This is a helper method of other methods that are called when instantiating the class."""

        df_full['time_off'] = 0
        for i in range(len(df_full)):
            zero = 0
            if (df_full['recur'][i] == 1):
                for c in range(len(df_full.values[0]) - 2):  # without churn and recur
                    if df_full.values[i][c] != 0:
                        for buy in range(c, c + len(df_full.values[i][c:-3])):
                            if (df_full.values[i][buy] == 0 and df_full.values[i][buy + 1] == 0):
                                zero += 1
                            if (df_full.values[i][buy] == 0 and df_full.values[i][buy + 1] == 0 and df_full.values[i][
                                buy + 2] != 0):
                                break
                        df_full.iloc[i, -1] = zero + 1
                        break
        return df_full


    def _add_churn_recur_timeoff(self):
        """Classify the data with chursn, recurrence, and time_off for each customer type.
        The winter and summer customers will have considered two winter or two summer harvests (2 years) 
        without purchases to classify a positive churn, while to a fullyear customer will have considered 
        one winter and one summer (1 year) without purchases."""
        
        # summer
        df_summer = self.df_summer.reset_index()
        df_summer['DATA_SAFRA'] = df_summer['DATA_SAFRA'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_summer = df_summer.set_index(['DATA_SAFRA'])
        df_summer.columns.name = None
        df_summer = df_summer.T
        df_summer_churn = self.add_churn_recur_values(df_summer)
        self.df_summer_churn = self.count_time_off(df_summer_churn)
        # winter
        df_winter = self.df_winter.reset_index()
        df_winter['DATA_SAFRA'] = df_winter['DATA_SAFRA'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_winter = df_winter.set_index(['DATA_SAFRA'])
        df_winter.columns.name = None
        df_winter = df_winter.T
        df_winter_churn = self.add_churn_recur_values(df_winter)
        self.df_winter_churn = self.count_time_off(df_winter_churn)
        # full-year
        df_fullyear = self.df_fullyear.reset_index()
        df_fullyear['DATA_SAFRA'] = df_fullyear['DATA_SAFRA'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_fullyear = df_fullyear.set_index(['DATA_SAFRA'])
        df_fullyear.columns.name = None
        df_fullyear = df_fullyear.T
        df_fullyear_churn = self.add_churn_recur_values(df_fullyear)
        self.df_fullyear_churn = self.count_time_off(df_fullyear_churn)

        return self.df_summer_churn, self.df_winter_churn, self.df_fullyear_churn


    def concat_general_dfs(self, dfs_list):
        """Receives a list of DataFrames to concatenate.
        It is mandatory that every DataFrame has a column in common."""

        df_complete = pd.concat(dfs_list)
        df_complete = df_complete.fillna(0)
        return df_complete

    
    def _create_df_churn_complete(self):
        """It considers the current year and month to choose the last 11 harvests to the model.
        Between April and September, the winter crop of the current year is considered as the last crop in the churn/recurrence analysis;
        while after October the last crop is the summer of the following year and between January and march the last crop is the summer of
        the current year.
        Returns:
            df_churn_complete: DataFrame complete with columns of harvests, churn, recurrence, and time_off."""

        current_year = time.strftime('%Y')
        current_month = time.strftime('%m')

        df_churn_complete = self.concat_general_dfs(
                [self.df_summer_churn, self.df_winter_churn, self.df_fullyear_churn])

        if 4 <= int(current_month) and int(current_month) < 10:
            self.df_churn_complete = df_churn_complete[                 
                [f'{int(current_year)-5}-07-01', f'{int(current_year)-4}-01-01', f'{int(current_year)-4}-07-01', f'{int(current_year)-3}-01-01', 
                f'{int(current_year)-3}-07-01', f'{int(current_year)-2}-01-01', f'{int(current_year)-2}-07-01', f'{int(current_year)-1}-01-01', 
                f'{int(current_year)-1}-07-01', f'{int(current_year)}-01-01',f'{int(current_year)}-07-01', 'churn', 'recorr', 'tempo_off']]
         
        elif int(current_month) >= 10 :                          
            self.df_churn_complete = df_churn_complete[
                [f'{int(current_year)-5}-07-01', f'{int(current_year)-4}-01-01', f'{int(current_year)-4}-07-01', f'{int(current_year)-3}-01-01', f'{int(current_year)-3}-07-01',
                f'{int(current_year)-2}-01-01', f'{int(current_year)-2}-07-01', f'{int(current_year)-1}-01-01', f'{int(current_year)-1}-07-01', f'{int(current_year)}-01-01',
                f'{int(current_year)}-07-01', f'{int(current_year)+1}-01-01', 'churn', 'recorr', 'tempo_off']]
        
        elif int(current_month) < 4:                                                                 
            self.df_churn_complete = df_churn_complete[
                [f'{int(current_year)-6}-07-01', f'{int(current_year)-5}-01-01', f'{int(current_year)-5}-07-01', f'{int(current_year)-4}-01-01', f'{int(current_year)-4}-07-01',
                f'{int(current_year)-3}-01-01', f'{int(current_year)-3}-07-01', f'{int(current_year)-2}-01-01', f'{int(current_year)-2}-07-01', f'{int(current_year)-1}-01-01',
                f'{int(current_year)-1}-07-01', f'{int(current_year)}-01-01', 'churn', 'recorr', 'tempo_off']]

        return self.df_churn_complete


    def _create_df_input_model(self):
        """Concatenate the df_churn_complete and df_stat_commodities DataFrames transforming the event columns
        in only one containing the sum of the participations.
        Depending on the initial model_type parameter 'churn' or 'recur' adjusts the data for input in training
        of the respective models.
        Returns:
            df_input_model: DataFrame"""

        df_churn = self.df_churn_complete[['recur', 'time_off', 'churn']]
        df_churn.fillna(0, inplace=True)

        df_concat = pd.concat([df_churn, self.df_stat_commodities], axis=1)
    
        cols_event_description = ['EVENT 2016', 'EVENT 2017', 'EVENT 2018', 
                                  'EVENT 2019', 'EVENT 2020', 'EVENT 2021']

        for i in cols_event_description:
            df_concat.loc[df_concat[f'{i}'] != 0, f'{i}'] = 1
        df_concat['Event_participation'] = df_concat[cols_event_description][:].sum(axis=1)
                   
        # se houver, remove colunas com valores negativos
        for i in df_concat.columns:
            try:
                df_concat[[i]] = pd.to_numeric(df_concat[[i]], errors='ignore', downcast='float')
                df_concat[[i]].fillna(0)
            except:
                pass
       
        df_concat = df_concat[df_concat >= 0].fillna(0)
    
        if self.model_type == 'churn' :

            self.df_input_model = df_concat[['DEFENSIVES', 'SPECIAL FERTILIZERS', 'FOLIARY FERTILIZERS',
                                              'SOIL FERTILIZERS', 'SEEDS', 'SF-16/16', 'SF-16/17', 'SF-17/17',
                                              'SF-17/18', 'SF-18/18', 'SF-18/19', 'SF-19/19', 'SF-19/20', 'SF-20/20',
                                              'SF-20/21', 'SF-21/21', 'SF-21/22', 'FUTURE SALE AGREEMENT',
                                              'NEW OPERATION SALE', 'DIRECT SALES',
                                              'FUTURE SALES - SIMPLE BILLING','DIRECT SALES ORDER',
                                              'SALE ON ORDER - SHIPMENT', 'SALE PRD',
                                              'Event_participation', 'churn']]
        elif self.model_type == 'recorr':

            self.df_input_model = df_concat[['DEFENSIVES', 'SPECIAL FERTILIZERS', 'FOLIARY FERTILIZERS',
                                              'SOIL FERTILIZERS', 'SEEDS', 'SF-16/16', 'SF-16/17', 'SF-17/17',
                                              'SF-17/18', 'SF-18/18', 'SF-18/19', 'SF-19/19', 'SF-19/20', 'SF-20/20',
                                              'SF-20/21', 'SF-21/21', 'SF-21/22', 'FUTURE SALE AGREEMENT',
                                              'NEW OPERATION SALE', 'DIRECT SALES',
                                              'FUTURE SALES - SIMPLE BILLING','DIRECT SALES ORDER',
                                              'SALE ON ORDER - SHIPMENT', 'SALE PRD',
                                              'Event_participation', 'recorr']]

        return self.df_input_model