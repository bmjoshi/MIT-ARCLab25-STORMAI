import os
import torch
from torch import nn
import joblib
import pandas as pd
import numpy as np
import dill
from datetime import timedelta 
from swai.XGboost import XGBoostDensityModel  # Assuming your previous class is saved in this module
#from swai.RandomForestDensity import RandomForestDensityModel
class ParticipantModel(nn.Module):
    def __init__(self, model_folder, plot_trajectory=False):
        super().__init__()
        self.plot = plot_trajectory
        self.model = XGBoostDensityModel()  # Use the XGBoost model
        self.model.load_data(model_folder)  # Load trained models

    def forward(self, omni2_data, initial_state={}):
        """
        Compute the propagated states and predict density.
        """
        start_time = initial_state['Timestamp']
        timestamps = [start_time + i * timedelta(minutes=10) for i in range(432)]
        lat_array = np.zeros(432)* initial_state['Latitude (deg)']
        lon_array = np.zeros(432)* initial_state['Longitude (deg)']
        alt_array = np.ones(432) * initial_state['Altitude (km)']

        # Ensure altitude is in km
        if np.max(alt_array) > 1000:
            alt_array = alt_array / 1000.0

        # Prepare input data for the XGBoost model
        input_data = np.column_stack([timestamps, lat_array, lon_array, alt_array])
        
        # Predict density
        density = self.model.run(input_data, omni2_data)
        
        # Store results in DataFrame
        sample_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Latitude': lat_array,
            'Longitude': lon_array,
            'Altitude': alt_array,
            'Density': density
        })
        
        return sample_df

def generate_model(model_folder, model_name='xgb_v12'):
    """
    Generate and save the trained model.
    """
    model = ParticipantModel(model_folder)
    torch.save(model, f'{model_name}.pkl', pickle_module=dill)

def run_model(model_name='xgb_v12', 
              initial_states_file='temp/test/initial_states.csv', 
              omni2_path='temp/test/omni2/'):
    """
    Load the model and make predictions based on initial states.
    """
    model = torch.load(f'{model_name}.pkl', pickle_module=dill)
    initial_states = pd.read_csv(initial_states_file)
    initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'])
    
    predictions = {}
    for _, row in initial_states.iterrows():
        file_id = row['File ID']
        omni2_data = pd.read_csv(os.path.join(omni2_path, f"omni2-{file_id}.csv"))
        omni2_data['time'] = pd.to_datetime(omni2_data['Timestamp'])
        
        result = model(omni2_data.ffill(), initial_state=row.drop("File ID").to_dict())
        print(result)
        predictions[file_id] = {
           # "Timestamp": result["Timestamp"].dt.isoformat().tolist(),
            "Orbit Mean Density (kg/m^3)": result["Density"].tolist()
        }
    
    return predictions

if __name__ == '__main__':
    generate_model('./XGBOOST')
    #run_model()
