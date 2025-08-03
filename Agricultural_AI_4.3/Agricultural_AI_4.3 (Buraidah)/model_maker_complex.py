import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings

warnings.filterwarnings('ignore')


class CropModelTrainer:
    def __init__(self):
        self.models = {}
        # CORRECTED: vpd_sensitivity is now included for each crop
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

    def load_weather_data(self, filename='Buraidah.csv'):
        try:
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Loaded {filename}: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"❌ Error: Data file '{filename}' not found!")
            return None

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

            if len(growing_season_data) >= 30:
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

    def generate_yields(self, features_df, crop_type):
        # CORRECTED: This function now properly uses vpd_sensitivity
        crop_params = self.crop_parameters[crop_type]
        base_yield = crop_params['base_yield']
        yields = []
        for _, row in features_df.iterrows():
            yield_factor = row['optimal_temp_ratio'] * min(1.0, row['total_precipitation'] / crop_params[
                'water_requirement'])

            vpd_sensitivity = crop_params['vpd_sensitivity']
            vpd_effect = 1 - (row['avg_vpd_max'] / 6) * vpd_sensitivity
            yield_factor *= max(0, vpd_effect)

            yields.append(base_yield * yield_factor)
        return np.array(yields)

    def train_and_store_model(self, features_df, yields, crop_type):
        print(f"\n🚀 Training model for {crop_type}...")

        core_features = [
            'total_precipitation', 'avg_temp_max', 'optimal_temp_ratio', 'total_gdd',
            'total_et0', 'total_radiation', 'avg_soil_temp', 'avg_vpd_max'
        ]
        X = features_df[core_features].fillna(0)
        y = yields

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        if crop_type == 'barley':
            print("   Tuning hyperparameters for Barley...")
            param_grid = {
                'n_estimators': [100, 150], 'max_depth': [5, 10, None],
                'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]
            }
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_scaled, y)
            print(f"   Best parameters found: {grid_search.best_params_}")
            model = grid_search.best_estimator_
            cv_score = grid_search.best_score_
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y)
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            cv_score = cv_scores.mean()

        print(f"   Cross-Validation R²: {cv_score:.4f}")

        if cv_score > 0.5:
            print(f"   ✅ Model for {crop_type} is reliable and will be saved.")
            self.models[crop_type] = {'model': model, 'scaler': scaler, 'features': core_features}
        else:
            print(f"   ❌ Model for {crop_type} is not reliable and will NOT be saved.")

    def save_models(self, filename='crop_predictor.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"\n✅ All reliable models have been saved to '{filename}'")


def main():
    trainer = CropModelTrainer()
    historical_weather_df = trainer.load_weather_data(filename='Buraidah.csv')
    if historical_weather_df is None:
        return

    crops = ['wheat', 'dates', 'tomatoes', 'barley', 'moringa', 'okra']
    for crop in crops:
        features_df = trainer.create_crop_features(historical_weather_df, crop)
        yields = trainer.generate_yields(features_df, crop)
        trainer.train_and_store_model(features_df, yields, crop)

    trainer.save_models()


if __name__ == "__main__":
    main()