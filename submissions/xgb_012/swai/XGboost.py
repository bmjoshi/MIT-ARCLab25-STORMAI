import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import logging
from joblib import Parallel, delayed
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.WARNING)

class XGBoostDensityModel:

    def __init__(self):
        # Initialize the model dictionary
        self.models = {}
        self.sw_data = None

        # Define column ranges for filtering
        self.col_ranges = {
            'altitude': (0, 1000),
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'average_ap_index_nT': (0, 400),
            'average_f10.7_index': (0, 200),
            'average_Lyman_alpha': (0, 0.1),
            'std_ap_index_nT': (0, 400),
            'std_f10.7_index': (0, 200),
            'std_Lyman_alpha': (0, 0.1),
            'min_ap_index_nT': (0, 400),
            'min_f10.7_index': (0, 200),
            'min_Lyman_alpha': (0, 0.1),
            'max_ap_index_nT': (0, 400),
            'max_f10.7_index': (0, 200),
            'max_Lyman_alpha': (0, 0.1)
        }

        # Define AP index bins and labels
        self.ap_bins = [0, 32, 65, 95, 156, 302, float('inf')]
        self.ap_labels = ["0-32", "33-65", "66-94", "95-155", "156-301", ">302"]

        # Define altitude bins and labels
        self.altitude_bins = [0, 200, 400, 600, 800, 1000]
        self.altitude_labels = ['0-200', '201-400', '401-600', '601-800', '801-1000']

        self.scale_factor = 1e12  # Scaling factor used in training

    def load_data(self, model_folder):
        """
        Load all models during initialization.
        """
        for ap_bin in self.ap_labels:
            for alt_bin in self.altitude_labels:
                model_path = os.path.join(
                    model_folder,
                    f"AP_{ap_bin}_Alt_{alt_bin}",
                    "xgboost_model.pkl"  # Changed from random_forest_model.pkl to xgboost_model.pkl
                )
                if os.path.exists(model_path):
                    self.models[(ap_bin, alt_bin)] = joblib.load(model_path)
                else:
                    logging.warning(f"Model not found for AP bin {ap_bin} and Altitude bin {alt_bin}.")

    def _validate_input(self, input_data):
        """
        Validate input data ranges.
        """
        for col, (min_val, max_val) in self.col_ranges.items():
            if col in input_data.columns:
                if not input_data[col].between(min_val, max_val).all():
                    logging.warning(f"Input data for {col} is out of range.")

    def _calculate_statistics(self, filtered_data):
        """
        Calculate statistics for space weather data.
        """
        averages = filtered_data[['ap_index_nT', 'f10.7_index', 'Lyman_alpha']].mean()
        std = filtered_data[['ap_index_nT', 'f10.7_index', 'Lyman_alpha']].std()
        min_vals = filtered_data[['ap_index_nT', 'f10.7_index', 'Lyman_alpha']].min()
        max_vals = filtered_data[['ap_index_nT', 'f10.7_index', 'Lyman_alpha']].max()
        return averages, std, min_vals, max_vals

    def _find_nearest_alt_bin(self, alt_value):
        """
        Find the nearest altitude bin for a given altitude value.
        """
        bin_midpoints = [(self.altitude_bins[i] + self.altitude_bins[i + 1]) / 2 for i in range(len(self.altitude_bins) - 1)]
        bin_midpoints.append(self.altitude_bins[-1])  # Add the last bin edge
        nearest_index = np.argmin(np.abs(np.array(bin_midpoints) - alt_value))
        return self.altitude_labels[nearest_index]

    def _find_available_model(self, ap_bin, alt_bin):
        """
        Find the nearest available model for a given AP bin and altitude bin.
        """
        if (ap_bin, alt_bin) in self.models:
            return self.models[(ap_bin, alt_bin)]
        for bin_label in self.altitude_labels:
            if (ap_bin, bin_label) in self.models:
                return self.models[(ap_bin, bin_label)]
        return None

    def run(self, input_data, sw_data):
        """
        Run the model to predict thermospheric density at a given time and multiple locations.
        """
        # Assume input_data is a numpy array of shape (n_samples, 4): [datetime, lat, lon, alt]
        time = input_data[:, 0]  # Already datetime
        lat = input_data[:, 1]
        lon = input_data[:, 2]
        alt = input_data[:, 3]

        if np.max(alt) > 1000:
        #logging.warning("Altitude values appear to be in meters, converting to kilometers.")
            alt = alt / 1000.0

        dt = time[0]
        start_time = dt - timedelta(days=5)

        # Load space weather data
        self.sw_data = sw_data
        self.sw_data['time'] = pd.to_datetime(self.sw_data.YEAR, format='%Y') + \
                            pd.to_timedelta(self.sw_data.DOY * 24 + self.sw_data.Hour, unit='hour')

        # Filter rows within 5-day range
        filtered_data = self.sw_data[(self.sw_data['time'] >= start_time) &
                                    (self.sw_data['time'] < dt)]

        # Calculate statistics
        averages, std, min_vals, max_vals = self._calculate_statistics(filtered_data)

        # Prepare input DataFrame
        inputs = pd.DataFrame({
            'altitude': alt,
            'longitude': lon,
            'latitude': lat,
            'average_ap_index_nT': averages['ap_index_nT'],
            'average_f10.7_index': averages['f10.7_index'],
            'average_Lyman_alpha': averages['Lyman_alpha'],
            'std_ap_index_nT': std['ap_index_nT'],
            'std_f10.7_index': std['f10.7_index'],
            'std_Lyman_alpha': std['Lyman_alpha'],
            'min_ap_index_nT': min_vals['ap_index_nT'],
            'min_f10.7_index': min_vals['f10.7_index'],
            'min_Lyman_alpha': min_vals['Lyman_alpha'],
            'max_ap_index_nT': max_vals['ap_index_nT'],
            'max_f10.7_index': max_vals['f10.7_index'],
            'max_Lyman_alpha': max_vals['Lyman_alpha'],
        })

        # Validate input data
        self._validate_input(inputs)

        # Determine AP bin
        ap_value = filtered_data['ap_index_nT'].iloc[-1]
        ap_bin = pd.cut([ap_value], bins=self.ap_bins, labels=self.ap_labels, include_lowest=True)[0]

        # Bin altitude values
        alt_bins = pd.cut(alt, bins=self.altitude_bins, labels=self.altitude_labels, include_lowest=True)

        # Results array
        results = np.zeros_like(alt, dtype=float)

        # Predict for each altitude bin
        for alt_bin in self.altitude_labels:
            mask = (alt_bins == alt_bin)
            if mask.any():
                model = self._find_available_model(ap_bin, alt_bin)
                if model is not None:
                    input_rows = inputs[mask].reindex(columns=model.feature_names_in_, fill_value=0)
                    input_rows = input_rows.apply(pd.to_numeric, errors='coerce')
                    results[mask] = model.predict(input_rows) / self.scale_factor
                else:
                    results[mask] = np.nan

        return results


def main():
    # Define paths
    model_folder = "/home/archana-trivedi/Documents/Work_Done/selenium/XGBOOST"
    path_to_swdata = "/home/archana-trivedi/Documents/codebench/submission_template_sgp4-main/submission_template_sgp4/test/omni2/omni2-664.csv"

    # Initialize and load the model
    model = XGBoostDensityModel()
    model.load_data(model_folder)

    # Load space weather data from CSV
    sw_data = pd.read_csv(path_to_swdata)

    # Define input data (time, latitude, longitude, altitude)
    dt = datetime(2003, 6, 28)
    input_data = np.array([
        [dt, 0, 0, 100],   # Example 1: Altitude 100 km
        [dt, 0, 0, 300],   # Example 2: Altitude 300 km
        [dt, 0, 0, 500],   # Example 3: Altitude 500 km
        [dt, 0, 0, 700],   # Example 4: Altitude 700 km
        [dt, 0, 0, 900]    # Example 5: Altitude 900 km
    ])

    # Run the model for a specific time and location
    density = model.run(input_data, sw_data)
    print(f"Predicted Density: {density}")
if __name__ == "__main__":
    main()
