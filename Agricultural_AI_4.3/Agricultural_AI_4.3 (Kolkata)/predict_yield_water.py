import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

class IrrigatedCropPredictor:
    def __init__(self, model_path='crop_predictor.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            print(f"✅ Models loaded successfully from '{model_path}'")
        except FileNotFoundError:
            print(f"❌ Error: Model file '{model_path}' not found. Please run the training script first.")
            self.models = {}

        # CORRECTED: Dictionary now matches the training script
        self.crop_parameters = {
            'wheat': {
                'optimal_temp_min': 15, 'optimal_temp_max': 25, 'water_requirement': 550,
                'growing_season_months': [11, 12, 1, 2, 3], 'base_yield': 3.0, 'vpd_sensitivity': 0.3
            },
            'barley': {
                'optimal_temp_min': 12, 'optimal_temp_max': 21, 'water_requirement': 400,
                'growing_season_months': [11, 12, 1, 2, 3], 'base_yield': 4.0, 'vpd_sensitivity': 0.25
            },
            'dates': {
                'optimal_temp_min': 25, 'optimal_temp_max': 38, 'water_requirement': 1500,
                'growing_season_months': list(range(1, 13)), 'base_yield': 6.0, 'vpd_sensitivity': 0.1
            },
            'tomatoes': {
                'optimal_temp_min': 18, 'optimal_temp_max': 29, 'water_requirement': 600,
                'growing_season_months': [10, 11, 12, 1, 2, 3], 'base_yield': 80.0, 'vpd_sensitivity': 0.4
            },
            'moringa': {
                'optimal_temp_min': 22, 'optimal_temp_max': 35, 'water_requirement': 600,
                'growing_season_months': list(range(1, 13)), 'base_yield': 15.0, 'vpd_sensitivity': 0.15
            },
            'okra': {
                'optimal_temp_min': 24, 'optimal_temp_max': 32, 'water_requirement': 500,
                'growing_season_months': [3, 4, 5, 6, 7, 8, 9, 10], 'base_yield': 15.0, 'vpd_sensitivity': 0.2
            }
        }

    def create_crop_features(self, weather_df, crop_type):
        df = weather_df.copy()
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        crop_params = self.crop_parameters[crop_type]

        df['temp_stress_cold'] = np.maximum(0, crop_params['optimal_temp_min'] - df['temperature_2m_min'])
        df['temp_stress_hot'] = np.maximum(0, df['temperature_2m_max'] - crop_params['optimal_temp_max'])
        temp_range = crop_params['optimal_temp_max'] - crop_params['optimal_temp_min']
        df['temp_optimality'] = np.where(
            (df['temperature_2m_min'] >= crop_params['optimal_temp_min']) & (
                    df['temperature_2m_max'] <= crop_params['optimal_temp_max']),
            1.0, np.maximum(0, 1 - (df['temp_stress_cold'] + df['temp_stress_hot']) / temp_range))

        base_temp = crop_params['optimal_temp_min']
        df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        df['gdd'] = np.maximum(0, df['temp_mean'] - base_temp)

        seasonal_features = []
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            growing_season_data = year_data[year_data['month'].isin(crop_params['growing_season_months'])]

            if len(growing_season_data) > 0:
                seasonal_row = {
                    'year': year,
                    'avg_temp_max': growing_season_data['temperature_2m_max'].mean(),
                    'optimal_temp_ratio': growing_season_data['temp_optimality'].mean(),
                    'total_precipitation': growing_season_data['precipitation_sum'].sum(),
                    'total_gdd': growing_season_data['gdd'].sum(),
                    'total_et0': growing_season_data['et0_fao_evapotranspiration'].sum(),
                    'total_radiation': growing_season_data['shortwave_radiation_sum'].sum(),
                    'avg_soil_temp': growing_season_data['soil_temperature_0_to_7cm_mean'].mean(),
                    'avg_vpd_max': growing_season_data['vapour_pressure_deficit_max'].mean(),
                }
                seasonal_features.append(seasonal_row)
        return pd.DataFrame(seasonal_features)

    def predict(self, future_weather_df):
        print("\n--- PREDICTING FUTURE YIELDS (WITH PERFECT IRRIGATION) ---")

        if not self.models:
            print("  No reliable models were loaded. Cannot make predictions.")
            return

        for crop_type in self.crop_parameters.keys():
             if crop_type in self.models:
                print(f"\nAnalyzing {crop_type.upper()}...")
                model_data = self.models[crop_type]
                features_df = self.create_crop_features(future_weather_df, crop_type)

                if len(features_df) == 0:
                    print(f"  ❌ Could not generate features for the future period.")
                    continue

                water_requirement = self.crop_parameters[crop_type]['water_requirement']
                forecasted_precipitation = features_df['total_precipitation'].iloc[0]
                water_needed_mm = max(0, water_requirement - forecasted_precipitation)
                liters_per_hectare = water_needed_mm * 10000

                features_for_prediction = features_df.copy()
                features_for_prediction['total_precipitation'] = water_requirement

                X_future = features_for_prediction[model_data['features']].fillna(0)
                scaler = model_data['scaler']
                model = model_data['model']

                X_future_scaled = scaler.transform(X_future)
                prediction = model.predict(X_future_scaled)
                predicted_yield = prediction[0]

                base_yield = self.crop_parameters[crop_type]['base_yield']
                performance_vs_perfect = (predicted_yield / base_yield) * 100
                year = features_df['year'].iloc[0]

                print(f"  💧 Irrigation Water Needed: {water_needed_mm:.1f} mm for the growing season.")
                print(f"     (Equivalent to {liters_per_hectare:,.0f} liters per hectare)")
                print(f"  🌱 Predicted Irrigated Yield for {crop_type} in {year}: {predicted_yield:.2f} tons/ha")
                print(f"     (This is {performance_vs_perfect:.1f}% of the perfect yield of {base_yield:.2f} tons/ha)")
             else:
                 print(f"\nSkipping {crop_type.upper()}: No reliable model was saved for this crop.")

def main():
    try:
        future_weather_df = pd.read_csv('future_weather_data.csv')
        future_weather_df['date'] = pd.to_datetime(future_weather_df['date'])
    except FileNotFoundError:
        print("❌ Error: 'future_weather_data.csv' not found. Please create this file first.")
        return

    predictor = IrrigatedCropPredictor()
    predictor.predict(future_weather_df)

if __name__ == "__main__":
    main()