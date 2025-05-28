from swai.ConditionalXGBModel import ConditionalXGBModel
from swai.ForecastingModel import ForecastingModel

class ConditionalHybridModel:
    def __init__(self):
        self.density_estimator = ConditionalXGBModel()
        self.forecaster = ForecastingModel()

        bin_map = {}
        bin_map['alt-0-f10-0'] = {'path_to_model': 'xgb-model-010-alt-0-f10-0.json'}
        bin_map['alt-2-f10-0'] = {'path_to_model': 'xgb-model-010-alt-2-f10-0.json'}
        bin_map['alt-2-f10-1'] = {'path_to_model': 'xgb-model-010-alt-2-f10-1.json'}
        bin_map['alt-2-f10-2'] = {'path_to_model': 'xgb-model-010-alt-2-f10-2.json'}
        bin_map['alt-3-f10-0'] = {'path_to_model': 'xgb-model-010-alt-3-f10-0.json'}
        bin_map['alt-3-f10-1'] = {'path_to_model': 'xgb-model-010-alt-3-f10-1.json'}
        bin_map['alt-3-f10-2'] = {'path_to_model': 'xgb-model-010-alt-3-f10-2.json'}
        bin_map['alt-4-f10-0'] = {'path_to_model': 'xgb-model-010-alt-4-f10-0.json'}
        bin_map['alt-4-f10-1'] = {'path_to_model': 'xgb-model-010-alt-4-f10-1.json'}
        bin_map['alt-4-f10-2'] = {'path_to_model': 'xgb-model-010-alt-4-f10-2.json'}

        self.density_estimator.load_model(bin_map)
        self.forecaster.load_model('tslanet_v5.pkl')

    def run(self, input_data, sw_data):

        base_val = self.density_estimator.run(input_data, sw_data)
        trend    = self.forecaster.run(input_data, sw_data)

        # combine two values to forecast the time series
        output = (1+trend)*base_val

        return output
