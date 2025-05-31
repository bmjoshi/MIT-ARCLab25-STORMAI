from xgboost import XGBRegressor

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

mu_km = 3.986004418e5  # Gravitational parameter for Earth (m^3/s^2)

class ConditionalXGBModel:

    def __init__(self):
        self.model = None
        
        self.features = []
        self.proxy_prefix = ['average', 'min', 'max']
        self.proxy_vars = ['ap_index_nT', 'f10.7_index', 'Lyman_alpha', 'Dst_index_nT', 'Lyman_alpha2','f10.7_index2', 'Lyman_alpha_f10.7', 'ap_index_nT2', 'ap_index_nT_f10.7']

        for p in self.proxy_prefix:
            for f in self.proxy_vars:
                varname = '{}_{}'.format(p, f)
                self.features.append(varname)
        
        self.features += ['altitude']

        self.col_ranges = {
            'altitude': (0, 1000),

            'average_ap_index_nT': (0,400),
            'average_f10.7_index': (63.4,250),
            'average_Lyman_alpha': (0.00588,0.010944),
            'average_Dst_index_nT': (-422,71),
            'average_Lyman_alpha2': (0, 1e-4),
            'average_f10.7_index2': (3969, 62500),
            'average_Lyman_alpha_f10.7': (0, 2.75),
            'average_ap_index_nT2': (0, 1.6e5),
            'average_ap_index_nT_f10.7': (0, 1e5),

            'min_ap_index_nT': (0,400),
            'min_f10.7_index': (63.4,250),
            'min_Lyman_alpha': (0.00588,0.010944),
            'min_Dst_index_nT': (-422,71),
            'min_Lyman_alpha2': (0, 1e-4),
            'min_f10.7_index2': (3969, 62500),
            'min_Lyman_alpha_f10.7': (0, 2.75),
            'min_ap_index_nT2': (0, 1.6e5),
            'min_ap_index_nT_f10.7': (0, 1e5),

            'max_ap_index_nT': (0,400),
            'max_f10.7_index': (63.4,250),
            'max_Lyman_alpha': (0.00588,0.010944),
            'max_Dst_index_nT': (-422,71),
            'max_Lyman_alpha2': (0, 1e-4),
            'max_f10.7_index2': (3969, 62500),
            'max_Lyman_alpha_f10.7': (0, 2.75),
            'max_ap_index_nT2': (0, 1.6e5),
            'max_ap_index_nT_f10.7': (0, 1e5),
        }

    
    def load_model(self, map):

        # check the map and load the models for different bins
        # The map would contain key: {path_to_model: ""}
        self.model = {}
        for xgbm in map:
            model_ = XGBRegressor()
            model_.load_model(map[xgbm]['path_to_model'])
            self.model[xgbm] = model_

    def run(self, input_data, sw_data):
        
        # eval conditions
        time = pd.to_datetime(input_data[:, 0])
        lat = input_data[:, 1]
        lon = input_data[:, 2]
        alt = input_data[:, 3]

        # Load the space weather data
        self.sw_data = sw_data
        self.sw_data['time'] = pd.to_datetime(self.sw_data.YEAR, format='%Y') + \
        pd.to_timedelta(self.sw_data.DOY * 24 + self.sw_data.Hour, unit='hour')
        self.sw_data['Lyman_alpha2'] = self.sw_data['Lyman_alpha']**2
        self.sw_data['f10.7_index2'] = self.sw_data['f10.7_index']**2
        self.sw_data['Lyman_alpha_f10.7'] = self.sw_data['f10.7_index']*self.sw_data['Lyman_alpha']
        self.sw_data['ap_index_nT2'] = self.sw_data['ap_index_nT']*self.sw_data['ap_index_nT']
        self.sw_data['ap_index_nT_f10.7'] = self.sw_data['f10.7_index']*self.sw_data['ap_index_nT']

        dt = time[0]
        start_time = dt - timedelta(days=5)

        # Filter rows from File2 within the 5-day range
        filtered_data = self.sw_data[(self.sw_data['time'] >= start_time) &
                                     (self.sw_data['time'] < dt)]

        # Calculate the average for the required columns
        stats = {}
        stats['average'] = filtered_data[self.proxy_vars].mean()
        stats['std'] = filtered_data[self.proxy_vars].std()
        stats['min'] = filtered_data[self.proxy_vars].min()
        stats['max'] = filtered_data[self.proxy_vars].max()


        inputs = {
            'altitude': alt,
            'longitude': lon,
            'latitude': lat
        }
        
        av_alt = np.mean(alt)

        for p in self.proxy_prefix:
            for f in self.proxy_vars:
                varname = '{}_{}'.format(p, f)
                inputs[varname] = stats[p][f]*np.ones_like(alt)
        
        inputs = pd.DataFrame(inputs)
        
        for col_ in self.col_ranges:
           mm = self.col_ranges[col_]
           inputs[col_] = (inputs[col_]-mm[0])/(mm[1]-mm[0])

        inputs = inputs[self.features].to_numpy()
        
        f10 = stats['average']['f10.7_index']

        output = np.ones_like(alt)*1e-12

        if av_alt<340:
            output = self.model['alt-0-f10-0'].predict(inputs)
        elif av_alt>=340 and av_alt<410 and f10<110.5:
            output = self.model['alt-2-f10-0'].predict(inputs)
        elif av_alt>=340 and av_alt<410 and f10>=110.5 and f10<156.7:
            output = self.model['alt-2-f10-1'].predict(inputs)
        elif av_alt>=340 and av_alt<410 and f10>=156.7:
            output = self.model['alt-2-f10-2'].predict(inputs)
        elif av_alt>=410 and av_alt<480 and f10<110.5:
            output = self.model['alt-3-f10-0'].predict(inputs)
        elif av_alt>=410 and av_alt<480 and f10>=110.5 and f10<156.7:
            output = self.model['alt-3-f10-1'].predict(inputs)
        elif av_alt>=410 and av_alt<480 and f10>=156.7:
            output = self.model['alt-3-f10-2'].predict(inputs)
        elif av_alt>=480 and av_alt<550 and f10<110.5:
            output = self.model['alt-4-f10-0'].predict(inputs)
        elif av_alt>=480 and av_alt<550 and f10>=110.5 and f10<156.7:
            output = self.model['alt-4-f10-1'].predict(inputs)
        elif av_alt>=480 and f10>=156.7:
            output = self.model['alt-4-f10-2'].predict(inputs)

        output = self.get_mean_density(output*1e-11, av_alt)
        return output


    def get_mean_density(self, darray, alt):
        global mu_km
        T = np.sqrt(4*(np.pi**2/mu_km)*(6375+alt)**3)
        T = 94*60
        f = T/600-9
        darray = pd.Series(darray)
        orbit_mean_density = (darray[::-1].rolling(window=9).mean())[::-1]
        orbit_mean_density_offset = (darray[::-1].rolling(window=10).mean())[::-1]
        orbit_mean_density = (orbit_mean_density*(1-f)+f*orbit_mean_density_offset)
        for i in range(9):
            orbit_mean_density[len(darray)-1-i] = orbit_mean_density[len(darray)-10]
        return orbit_mean_density.to_list()



def main():

    # define the map
    map = {}
    map['alt-0-f10-0'] = {'path_to_model': 'models/xgb-model-011-alt-0-f10-0.json'}
    map['alt-2-f10-0'] = {'path_to_model': 'models/xgb-model-011-alt-2-f10-0.json'}
    map['alt-2-f10-1'] = {'path_to_model': 'models/xgb-model-011-alt-2-f10-1.json'}
    map['alt-2-f10-2'] = {'path_to_model': 'models/xgb-model-011-alt-2-f10-2.json'}
    map['alt-3-f10-0'] = {'path_to_model': 'models/xgb-model-011-alt-3-f10-0.json'}
    map['alt-3-f10-1'] = {'path_to_model': 'models/xgb-model-011-alt-3-f10-1.json'}
    map['alt-3-f10-2'] = {'path_to_model': 'models/xgb-model-011-alt-3-f10-2.json'}
    map['alt-4-f10-0'] = {'path_to_model': 'models/xgb-model-011-alt-4-f10-0.json'}
    map['alt-4-f10-1'] = {'path_to_model': 'models/xgb-model-011-alt-4-f10-1.json'}
    map['alt-4-f10-2'] = {'path_to_model': 'models/xgb-model-011-alt-4-f10-2.json'}
    

    model = ConditionalXGBModel()
    model.load_model(map) #, path_to_scaler)

    # Run the model for a specific time and input data
    sample_df = pd.DataFrame({
        'Timestamp': ['2001-01-01 00:00:00', '2001-02-01 00:00:00'],
        'Latitude': [0, 0],
        'Longitude': [0, 0],
        'Altitude': [400, 430]
    })
    lla_array = sample_df[['Timestamp', 'Latitude', 'Longitude', 'Altitude']].to_numpy()
    density = model.run(lla_array)
    print(f"Predicted Density: {density}")


if __name__ == "__main__":
    main()
