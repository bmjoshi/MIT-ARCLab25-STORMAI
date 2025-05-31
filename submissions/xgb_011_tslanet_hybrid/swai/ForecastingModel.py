import numpy as np
import pandas as pd

import torch
from torch import nn

import numpy as np
import math
import pickle

from swai.TSLANet import TSLANet

class ForecastingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration
        self.in_channels = 19  # Univariate time series
        self.patch_size = 48
        self.embed_dim = 64
        self.num_layers = 5
        self.num_classes = None  # For classification
        self.forecast_horizon = 432  # For forecasting

    def load_model(self, model_path='tslanet_v1.pkl'):
        self.model = TSLANet(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            forecast_horizon=self.forecast_horizon
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))

    def run(self, input_data, sw_data):

        # eval conditions
        time = pd.to_datetime(input_data[:, 0])
        lat = input_data[:, 1]
        lon = input_data[:, 2]
        alt = input_data[:, 3]

        self.col_ranges = {
        'altitude': (0,1000),
        'ap_index_nT': (0,400),
        'f10.7_index': (63.4,250),
        'Lyman_alpha': (0.00588,0.010944),
        'Dst_index_nT': (-422,71),
        'BX_nT_GSE_GSM': (-40.8,34.8),
        'BY_nT_GSE': (-33.2,63.4),
        'BZ_nT_GSE': (-53.7,37.5),
        'SW_Proton_Density_N_cm3': (0.1,137.2),
        'SW_Plasma_Speed_km_s': (233,1189),
        'Magnetosonic_Mach_number': (0.6,14.3),
        'log_Lyman_alpha2': (4, 8),
        'f10.7_index2': (3969, 62500),
        'Lyman_alpha_f10.7': (0, 2.75),
        'ap_index_nT2': (0, 1.6e5),
        'ap_index_nT_f10.7': (0, 1e5),
        'log_xrsb_flux': (2.5, 5),
        'log_xrsb_flux2': (5,10),
        'log_xrsb_flux_Lyman_alpha': (5, 9)
        }
        
        self.sw_varlist = ['ap_index_nT', 'f10.7_index', 'Lyman_alpha', 'Dst_index_nT',
                'BX_nT_GSE_GSM', 'BY_nT_GSE', 'BZ_nT_GSE', 'SW_Proton_Density_N_cm3',
                'SW_Plasma_Speed_km_s', 'Magnetosonic_Mach_number', 'Lyman_alpha2',
                'f10.7_index2', 'Lyman_alpha_f10.7', 'ap_index_nT2', 'ap_index_nT_f10.7',
                'xrsb_flux', 'xrsb_flux2', 'xrsb_flux_Lyman_alpha']
        
        # features for the model
        self.features = ['altitude', 'ap_index_nT', 'f10.7_index', 'Lyman_alpha', 'Dst_index_nT',
                'BX_nT_GSE_GSM', 'BY_nT_GSE', 'BZ_nT_GSE', 'SW_Proton_Density_N_cm3',
                'SW_Plasma_Speed_km_s', 'Magnetosonic_Mach_number', 'log_Lyman_alpha2',
                'f10.7_index2', 'Lyman_alpha_f10.7', 'ap_index_nT2', 'ap_index_nT_f10.7',
                'log_xrsb_flux', 'log_xrsb_flux2', 'log_xrsb_flux_Lyman_alpha']

        x = sw_data[self.sw_varlist].to_numpy()

        # convert space-weather variables to log scale
        log_varlist = ['Lyman_alpha2', 'xrsb_flux', 'xrsb_flux2', 'xrsb_flux_Lyman_alpha']
        for v in log_varlist:
            idx = self.sw_varlist.index(v)
            mask = (x[:,idx]==0)
            x[:,idx] = x[:,idx]+(mask)*(10**(-self.col_ranges['log_'+v][0]))
            x[:,idx] = -np.log10(x[:,idx])

        # Add altitudes form density data to SW data
        x = np.concatenate((alt[0]*np.ones(x.shape[0]).T.reshape(x.shape[0], 1), x), axis=1)

        # scale the features
        for idx, v in enumerate(self.features):
            x[:,idx] = (x[:,idx] - self.col_ranges[v][0]) / (self.col_ranges[v][1] - self.col_ranges[v][0])
        

        # get the time series
        x = torch.from_numpy(x).float().permute(1, 0)
        x = x.unsqueeze(0)
        x = x.to(self.device)

        self.model.eval()
        output = None
        with torch.no_grad():
            output = self.model(x)

        tarray = np.arange(432)

        trend = output['forecasting'].cpu().numpy().flatten()
        return trend
