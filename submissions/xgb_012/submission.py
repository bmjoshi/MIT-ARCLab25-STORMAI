import torch
from pathlib import Path
import dill
import time
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
import json
import numpy as np
import pandas as pd
# from propagator import prop_orbit
#from gppropagator import prop_orbit

# Paths
TRAINED_MODEL_PATH = Path('xgb_v12.pkl')
#TEST_DATA_DIR = os.path.join('/app', 'data', 'dataset', 'test')
#initial=os.path.join('/app','input_data')
#initial_states_file = os.path.join(initial, "initial_states.csv")
#TEST_PREDS_FP = Path('/app/output/prediction.json')

path_prefix    = '../../local_test/'
TEST_DATA_DIR  = f'/home/oem/Work/mit-arclab-challenge/data/train_cases/'
TEST_PREDS_FP  = f'{path_prefix}/predictions/train_cases_xgb_v12/'

if not os.path.exists(TEST_PREDS_FP):
   os.mkdir(f'{TEST_PREDS_FP}')

TEST_PREDS_FP += 'predictions.json'
initial_states_file = os.path.join(TEST_DATA_DIR, "initial_states.csv")
omni2_path = os.path.join(TEST_DATA_DIR, "omni2")

# Load initial states
initial_states = pd.read_csv(initial_states_file) # , usecols=['File ID', 'Timestamp', 'Semi-major Axis (km)', 'Eccentricity', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)'])
initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'], format='mixed')

# Load the trained model
start_time = time.time()
model = torch.load(f'{TRAINED_MODEL_PATH}', pickle_module=dill)
#print(f"Model loading time: {time.time() - start_time:.2f} seconds")

# Initialize predictions dictionary
predictions = {}

# Iterate over initial states
for _, row in initial_states.iterrows():
    file_id = row['File ID']
    #print(f"Processing file ID: {file_id}")

    # Load OMNI2 data
    start_time = time.time()
    omni2_file = os.path.join(omni2_path, f"omni2-{file_id}.csv")
    omni2_data = pd.read_csv(omni2_file)
    omni2_data['time'] = pd.to_datetime(omni2_data['Timestamp'], format='mixed')
    omni2_data = omni2_data.ffill()
    #print(f"OMNI2 data loading and preprocessing time: {time.time() - start_time:.2f} seconds")

    # Prepare initial state
    initial_state = row.drop("File ID")

    # Run the model
    start_time = time.time()
    result = model(omni2_data, initial_state=initial_state.to_dict())
    #print(f"Model execution time: {time.time() - start_time:.2f} seconds")

    # Process results
    start_time = time.time()
    result['Timestamp'] = pd.to_datetime(result['Timestamp'])
    predictions[file_id] = {
        "Timestamp": list(map(lambda ts: ts.isoformat(), result["Timestamp"])),
        "Orbit Mean Density (kg/m^3)": result["Density"].tolist()
    }
    
    #print(f"Result processing time: {time.time() - start_time:.2f} seconds")

    print(f"Total time for file ID {file_id}: {time.time() - start_time:.2f} seconds")

# Save predictions to JSON
start_time = time.time()
with open(TEST_PREDS_FP, "w") as outfile:
    json.dump(predictions, outfile)
print(f"Predictions saved to {TEST_PREDS_FP} in {time.time() - start_time:.2f} seconds")

print("Script execution completed.")
