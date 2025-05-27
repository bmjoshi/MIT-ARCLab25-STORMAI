import os
import torch
from torch import nn
#from pymsis import msis
import dill

import numpy as np
import pandas as pd

from swai.ConditionalXGBModel import ConditionalXGBModel

from gppropagator_vec import prop_orbit

class ParticipantModel(nn.Module):
    def __init__(self, plot_trajectory=False):
        super().__init__()
        plot = plot_trajectory

        self.swmodel = ConditionalXGBModel()

    def preprocess_omni(self, data):
        '''
        Check if the variables are within the range.
        If the values are outside bounds fill with nan values.
        '''
        bounds = {
            'ID_for_IMF_spacecraft':          9.900000e+01,
            'ID_for_SW_Plasma_spacecraft':    9.900000e+01,
            'num_points_IMF_averages':        9.990000e+02,
            'num_points_Plasma_averages':     9.990000e+02,
            'Scalar_B_nT':                    9.999000e+02,
            'Vector_B_Magnitude_nT':          9.999000e+02,
            'Lat_Angle_of_B_GSE':             9.999000e+02,
            'Long_Angle_of_B_GSE':            9.999000e+02,
            'BX_nT_GSE_GSM':                  9.999000e+02,
            'BY_nT_GSE':                      9.999000e+02,
            'BZ_nT_GSE':                      9.999000e+02,
            'BY_nT_GSM':                      9.999000e+02,
            'BZ_nT_GSM':                      9.999000e+02,
            'RMS_magnitude_nT':               9.999000e+02,
            'RMS_field_vector_nT':            9.999000e+02,
            'RMS_BX_GSE_nT':                  9.999000e+02,
            'RMS_BY_GSE_nT':                  9.999000e+02,
            'RMS_BZ_GSE_nT':                  9.999000e+02,
            'SW_Plasma_Temperature_K':        9.999999e+06,
            'SW_Proton_Density_N_cm3':        9.999000e+02,
            'SW_Plasma_Speed_km_s':           9.999000e+03,
            'SW_Plasma_flow_long_angle':      9.999000e+02,
            'SW_Plasma_flow_lat_angle':       9.999000e+02,
            'Alpha_Prot_ratio':               9.999000e+00,
            'sigma_T_K':                      9.999999e+06,
            'sigma_n_N_cm3':                  9.999000e+02,
            'sigma_V_km_s':                   9.999000e+03,
            'sigma_phi_V_degrees':            9.999000e+02,
            'sigma_theta_V_degrees':          9.999000e+02,
            'sigma_ratio':                    9.999000e+00,
            'Flow_pressure':                  9.999000e+01,
            'E_electric_field':               9.999900e+02,
            'Plasma_Beta':                    9.999900e+02,
            'Alfen_mach_number':              9.999000e+02,
            'Magnetosonic_Mach_number':       9.990000e+01,
            'Quasy_Invariant':                9.999900e+00,
            'Kp_index':                       9.000000e+01,
            #'f10.7_index':                   250,
            'f10.7_index':                    9.999000e+02,
            'AE_index_nT':                    9.999000e+03,
            'AL_index_nT':                    9.999900e+04,
            'AU_index_nT':                    9.999900e+04,
            'pc_index':                       9.999000e+02,
            'Proton_flux_>1_Mev':             9.999999e+05,
            'Proton_flux_>2_Mev':             9.999999e+04,
            'Proton_flux_>4_Mev':             9.999999e+04,
            'Proton_flux_>10_Mev':            9.999999e+04,
            'Proton_flux_>30_Mev':            9.999999e+04,
            'Proton_flux_>60_Mev':            9.999999e+04,
        }

        for var in bounds:
            data.loc[data[var]>=bounds[var], var] = np.nan
        
        return data


    def resample_data(self, data):
        """
        Resample the data to a specified frequency.
        """

        # If the DataFrame is empty, fill with 0
        if len(data)==0:
            null_df = pd.DataFrame({
                'Index': np.arange(1440)
                })
            for key in data:
                null_df[key] = np.zeros(1440)
            return null_df

        # Fill missing timestamps
        
        start_time = data['Timestamp'].min()
        end_time = data['Timestamp'].max()

        # Check if the Date Range if euqal to 1440 hours
        if (end_time - start_time).total_seconds() < 1440*60:
            print('Date Range is less than 1440 hours')

        all_times = pd.date_range(start=start_time, end=end_time, freq='1h')
        data = data.set_index('Timestamp').reindex(all_times)
        data.rename(columns={'index': 'Timestamp'}, inplace=True)

        data = data.resample('1h').mean()

        data.ffill(inplace=True)
        data.bfill(inplace=True)
        
        # Check if the length of the data is equal to 1440
        if len(data) > 1440:
            # remove extra data
            data = data.iloc[-1440:].copy()

        Index = np.arange(len(data))
        data['Index'] = Index
        data.set_index('Index', inplace=True)
        return data
    
    def forward(self, omni2_data, goes_data, initial_state={}):
        """
        Forward pass of the model.
        Args:
            omni2_data (pd.DataFrame): DataFrame containing OMNI2 data.
            goes_data (pd.DataFrame): DataFrame containing GOES data.
            initial_state (dict): Dictionary containing the initial state.
        Returns:
            pd.DataFrame: DataFrame containing the predicted density.
        """
        # get the propagated states from the initial state
        start_time = initial_state['Timestamp']
        timestamps = [start_time + i*pd.to_timedelta(10, unit='minutes') for i in range(432)]
        lat_array = np.zeros(432)
        lon_array = np.zeros(432)
        alt_array = np.ones(432)*initial_state['Altitude (km)']

        # check if the altitude is INF and use orbit propagator to generate
        # LLA values instead
        if initial_state['Altitude (km)']>1e4:
            timestamps, lat_array, lon_array, alt_array = prop_orbit(initial_state, None)

        # get the density for each state
        sample_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Latitude': lat_array,
            'Longitude': lon_array,
            'Altitude': alt_array
        })

        # Replace outliers from OMNI2 with NaN values
        omni2_data = self.preprocess_omni(omni2_data)

        # Combine OMNI2 and GOES data to form a common SW data
        goes_data = self.resample_data(goes_data)       
        omni2_data = self.resample_data(omni2_data)
        
        # Merge the dataframes on Index
        sw_data = pd.merge(omni2_data, goes_data, on='Index', how='left')

        # Quadratic terms
        sw_data['xrsb_flux2']   = sw_data['xrsb_flux']**2
        sw_data['Lyman_alpha2'] = sw_data['Lyman_alpha']**2
        sw_data['f10.7_index2'] = sw_data['f10.7_index']**2
        sw_data['ap_index_nT2'] = sw_data['ap_index_nT']*sw_data['ap_index_nT']

        # Cross terms
        sw_data['xrsb_flux_Lyman_alpha'] = sw_data['xrsb_flux']*sw_data['Lyman_alpha']
        sw_data['Lyman_alpha_f10.7']     = sw_data['f10.7_index']*sw_data['Lyman_alpha']
        sw_data['ap_index_nT_f10.7']     = sw_data['f10.7_index']*sw_data['ap_index_nT']    

        lla_array = sample_df[['Timestamp', 'Latitude', 'Longitude', 'Altitude']].to_numpy()
        density = self.swmodel.run(lla_array, sw_data)            
        sample_df['density'] = density
        return sample_df

def generate_model(model_name = 'xgb_v11'):
    model = ParticipantModel()

    # save the model with custom name
    torch.save(model, f'{model_name}.pkl', pickle_module=dill)

if __name__ == "__main__":
    generate_model()
