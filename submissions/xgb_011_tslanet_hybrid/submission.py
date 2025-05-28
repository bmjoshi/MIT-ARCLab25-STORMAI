import torch
from pathlib import Path
import dill
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

import json

from gppropagator_vec import prop_orbit

TRAINED_MODEL_PATH = Path('xgb_v11_tslanet_hybrid.pkl')
TEST_DATA_DIR = os.path.join('/app','data', 'dataset', 'test')
TEST_PREDS_FP = Path('/app/output/prediction.json')
initial_states_file = os.path.join('/app/input_data',"initial_states.csv")

#---------------------------------
# Local Testing
#---------------------------------

#path_prefix    = '../../local_test/'
#TEST_DATA_DIR  = f'/home/oem/Work/mit-arclab-challenge/data/phase1_test_cases/'
#TEST_PREDS_FP  = f'{path_prefix}/predictions/phase1_test_cases_tslanet_hybrid/'

#if not os.path.exists(TEST_PREDS_FP):
#    os.mkdir(f'{TEST_PREDS_FP}')

#TEST_PREDS_FP += 'predictions.json'
#initial_states_file = os.path.join(TEST_DATA_DIR, "initial_states.csv")
#---------------------------------

# Paths

omni2_path = os.path.join(TEST_DATA_DIR, "omni2")
goes_path = os.path.join(TEST_DATA_DIR, "goes")

#---------------------------------
model = torch.load(f'{TRAINED_MODEL_PATH}', pickle_module=dill)
predictions = {}
#---------------------------------

# Load initial states
initial_states = pd.read_csv(initial_states_file)
initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'], format='mixed')

omni2_features = ["YEAR","DOY","Hour","Bartels_rotation_number","ID_for_IMF_spacecraft",
                  "ID_for_SW_Plasma_spacecraft","num_points_IMF_averages",
                  "num_points_Plasma_averages","Scalar_B_nT","Vector_B_Magnitude_nT",
                  "Lat_Angle_of_B_GSE","Long_Angle_of_B_GSE","BX_nT_GSE_GSM","BY_nT_GSE","BZ_nT_GSE",
                  "BY_nT_GSM","BZ_nT_GSM","RMS_magnitude_nT","RMS_field_vector_nT","RMS_BX_GSE_nT",
                  "RMS_BY_GSE_nT","RMS_BZ_GSE_nT","SW_Plasma_Temperature_K","SW_Proton_Density_N_cm3",
                  "SW_Plasma_Speed_km_s","SW_Plasma_flow_long_angle","SW_Plasma_flow_lat_angle",
                  "Alpha_Prot_ratio","sigma_T_K","sigma_n_N_cm3","sigma_V_km_s",
                  "sigma_phi_V_degrees","sigma_theta_V_degrees","sigma_ratio",
                  "Flow_pressure","E_electric_field","Plasma_Beta","Alfen_mach_number",
                  "Magnetosonic_Mach_number","Quasy_Invariant","Kp_index","R_Sunspot_No",
                  "Dst_index_nT","ap_index_nT","f10.7_index","AE_index_nT","AL_index_nT",
                  "AU_index_nT","pc_index","Lyman_alpha","Proton_flux_>1_Mev",
                  "Proton_flux_>2_Mev","Proton_flux_>4_Mev","Proton_flux_>10_Mev",
                  "Proton_flux_>30_Mev","Proton_flux_>60_Mev","Flux_FLAG","Timestamp"]

goes_features = ['Timestamp', 'xrsb_flux']

for _, row in initial_states.iterrows():
    file_id = row['File ID']
    row['Altitude (km)'] = row['Altitude (km)']/1e3 # alt is in meters?!!, convert to km

    omni2_file = os.path.join(omni2_path, f"omni2-{file_id}.csv")
    goes_file = os.path.join(goes_path, f"goes-{file_id}.csv")

    omni2_data = pd.read_csv(omni2_file, usecols=omni2_features)
    omni2_data['Timestamp'] = pd.to_datetime(omni2_data['Timestamp'], format='mixed')

    goes_data = pd.read_csv(goes_file, usecols=goes_features)
    goes_data['Timestamp'] = pd.to_datetime(goes_data['Timestamp'], format='mixed')

    # Fill the missing data (if it exists)
    omni2_data.ffill(inplace=True)
    omni2_data.bfill(inplace=True)
    goes_data.ffill(inplace=True)
    goes_data.bfill(inplace=True)
    
    initial_state = row.drop("File ID")
    print(initial_state)
    result = model(omni2_data, goes_data, initial_state=initial_state.to_dict())
    result['Timestamp'] = pd.to_datetime(result['Timestamp'])

    predictions[file_id] = {
        "Timestamp": list(map(lambda ts: ts.isoformat(), result["Timestamp"])),
        "Orbit Mean Density (kg/m^3)": result["density"].tolist()
    }
    print(f"Model execution for {file_id} Finished")

with open(TEST_PREDS_FP, "w") as outfile: 
    json.dump(predictions, outfile, indent=4)

print("Saved predictions to: {}".format(TEST_PREDS_FP))
# time.sleep(360) # EVALAI BUG FIX
