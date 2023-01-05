import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb'))        
        self.store_type_encoder = pickle.load(open(self.home_path + 'parameters/store_type_encoder.pkl', 'rb'))
    
    def data_cleaning(self, df1):
        
        # ==============
        # Rename columns
        # ==============
        
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        
        # cols_old = list(df1.columns)
        
        snakecase = lambda x: inflection.underscore(x)
        
        cols_new = list(map(snakecase, cols_old))
        
        df1.columns = cols_new
        
        # ======================
        # Fillout Missing Values
        # ======================
        
        # object to datetime format
        df1['date'] = pd.to_datetime(df1['date'])

        # competition_distance
        max_dist = df1['competition_distance'].max()
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 3*max_dist if math.isnan(x) else x)

        # competition_open_since_month
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # promo2_since_week
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # fillout missing values with zero values
        df1['promo_interval'].fillna(0, inplace=True)
        
        # =================
        # Change data types
        # =================
        
        # float to int format
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype(int)

        # float to int format
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering(self, df2):
        
        # year
        df2['year'] = df2['date'].dt.year

        # day
        df2['day'] = df2['date'].dt.day

        # month
        df2['month'] = df2['date'].dt.month

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # year month
        df2['year_month'] = df2['date'].dt.strftime('%Y-%m')

        # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)

        # competition time month
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']).apply(lambda x: x.days)/30).astype(int)

        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w'))

        # promo time week
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']).apply(lambda x: x.days)/7).astype(int)

        # promo time day
        df2['promo_time_day'] = (df2['date'] - df2['promo_since']).apply(lambda x: x.days)

        # month_map
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df2['month_map'] = df2['date'].dt.month.map(month_map)

        # is_promo
        df2['is_promo'] = df2[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 
                                                                    'easter_holiday' if x == 'b' else 
                                                                    'christmas'      if x == 'c' else 'regular_day')
        
        # ===================
        # VARIABLES FILTERING
        # ===================
        
        # Filtering Rows        
        df2 = df2[df2['open'] != 0]
        
        # Filtering Columns
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2.drop(cols_drop, axis=1, inplace=True)
        
        return df2
    
    def data_preparation(self, df5):
        
        # =========
        # rescaling
        # =========
        
        # competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)
        
        # competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)
        
        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)
        
        # =========
        # Enconding 
        # =========
        
        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])
        
        # store_type encoder
        df5['store_type'] = self.store_type_encoder.transform(df5[['store_type']].values)

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)
        
        # =====================
        # Nature transformation
        # =====================
        
        # day of week
        freq_day_of_week = 2. * np.pi/7
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * freq_day_of_week))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * freq_day_of_week))

        # month
        freq_month = 2. * np.pi/12
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * freq_month))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * freq_month))

        # day 
        freq_day = 2. * np.pi/30
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * freq_day))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * freq_day))

        # week of year
        freq_week_of_year = 2. * np.pi/52
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * freq_week_of_year))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * freq_week_of_year))       
        
        # =================
        # Feature Selection
        # =================
        
        # boruta result to use
        cols_selected = [
            'store',
            'promo',
            'store_type',
            'assortment',
            'competition_distance',
            'competition_open_since_month',
            'competition_open_since_year',
            'promo2',
            'promo2_since_week',
            'promo2_since_year',
            'competition_time_month',
            'promo_time_week',
            'day_of_week_sin',
            'day_of_week_cos',
            'month_cos',
            'day_sin',
            'day_cos',
            'week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
    
        # prediction
        pred = model.predict(test_data)
        
        # add pred to original data
        original_data['prediction'] = np.exp(pred)
        
        return original_data.to_json(orient='records', date_format='iso')
