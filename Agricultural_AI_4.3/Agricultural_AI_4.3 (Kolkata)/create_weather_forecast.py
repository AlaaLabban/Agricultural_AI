import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class WeatherForecaster:
    def __init__(self, historical_data_path='Kolkata.csv'):
        """Loads the historical weather data."""
        try:
            self.df = pd.read_csv(historical_data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"✅ Loaded historical data: {self.df.shape}")
        except FileNotFoundError:
            print(f"❌ Error: Historical data file '{historical_data_path}' not found.")
            self.df = None

    def create_features(self, df):
        """Creates time-based and lag/rolling features."""
        df = df.copy()
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        for col in ['temperature_2m_max', 'temperature_2m_min', 'et0_fao_evapotranspiration', 'shortwave_radiation_sum',
                    'vapour_pressure_deficit_max']:
            if col in df.columns:
                df[f'{col}_lag_1'] = df[col].shift(1)

        for col in ['temperature_2m_max', 'temperature_2m_min']:
            if col in df.columns:
                df[f'{col}_rolling_7'] = df[col].rolling(window=7).mean()

        return df

    def create_evaluation_plots(self, y_true, y_pred, dates, target_name):
        """Generates and displays a panel of three evaluation plots."""
        print(f"    Generating evaluation plots for {target_name}...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f'Evaluation for: {target_name}', fontsize=16)

        axes[0].plot(dates, y_true, label='Actual', color='blue', alpha=0.7)
        axes[0].plot(dates, y_pred, label='Predicted', color='red', linestyle='--')
        axes[0].set_title('Actual vs. Predicted Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(target_name)
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1].set_title('Actual vs. Predicted Scatter')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        r2 = r2_score(y_true, y_pred)
        axes[1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1].transAxes, va='top', ha='left', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        residuals = y_true - y_pred
        axes[2].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
        axes[2].axhline(y=0, color='r', linestyle='--')
        axes[2].set_title('Residuals vs. Predicted')
        axes[2].set_xlabel('Predicted Values')
        axes[2].set_ylabel('Residuals')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
        print(f"    Displaying plots. Please close the plot window to continue.")

    def create_forecast(self):
        """
        Trains models for each weather variable, evaluates them, and predicts the next year.
        """
        if self.df is None: return
        print("\n--- Creating Weather Forecast for Next Year ---")

        historical_df_with_features = self.create_features(self.df.copy())

        all_feature_columns = [col for col in historical_df_with_features.columns if
                               col not in self.df.columns and col != 'date']

        historical_df_featured = historical_df_with_features.dropna(subset=all_feature_columns)

        features = [col for col in historical_df_featured.columns if col not in self.df.columns and col != 'date']

        X = historical_df_featured[features]
        original_target_cols = [col for col in self.df.columns if col != 'date']
        y_df = historical_df_featured[original_target_cols]

        X_train, X_test, y_train_df, y_test_df, dates_train, dates_test = train_test_split(
            X, y_df, historical_df_featured['date'], test_size=0.2, shuffle=False
        )

        last_date = self.df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=366)
        forecast_df = pd.DataFrame({'date': future_dates})

        target_variables = [
            'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
            'et0_fao_evapotranspiration', 'shortwave_radiation_sum',
            'soil_temperature_0_to_7cm_mean', 'vapour_pressure_deficit_max'
        ]

        for target in target_variables:
            if target not in self.df.columns:
                print(f"  ⚠️  Skipping {target}: Column not found in historical data.")
                forecast_df[target] = 0
                continue

            if target == 'precipitation_sum':
                print(f"\n  Generating heuristic forecast for {target}...")

                # --- MODIFIED: Dynamically find the rainy season ---
                monthly_rain = self.df.groupby(self.df['date'].dt.month)['precipitation_sum'].sum()
                # Define rainy season as months with > 5% of total annual rain
                rainy_season_months = monthly_rain[monthly_rain > monthly_rain.sum() * 0.05].index.tolist()
                print(f"    Dynamically determined rainy season months: {rainy_season_months}")

                rainy_days_hist = self.df[self.df['precipitation_sum'] > 0]
                avg_rain_days_per_year = len(rainy_days_hist) / len(self.df['date'].dt.year.unique())
                avg_total_precip_per_year = self.df.groupby(self.df['date'].dt.year)['precipitation_sum'].sum().mean()

                print(
                    f"    Historical average: {int(avg_rain_days_per_year)} rainy days per year, totaling {avg_total_precip_per_year:.2f}mm")

                future_precip = np.zeros(len(future_dates))
                future_df_temp = self.create_features(pd.DataFrame({'date': future_dates}))
                possible_rain_indices = future_df_temp[future_df_temp['month'].isin(rainy_season_months)].index

                # --- MODIFIED: Safeguard against the error ---
                num_rain_days = min(int(round(avg_rain_days_per_year)), len(possible_rain_indices))
                if num_rain_days < int(round(avg_rain_days_per_year)):
                    print(
                        f"    Warning: Not enough days in rainy season to match historical average. Using {num_rain_days} days.")

                if num_rain_days > 0:
                    rainy_day_indices = np.random.choice(possible_rain_indices, num_rain_days, replace=False)
                    random_amounts = np.random.rand(num_rain_days)
                    random_amounts /= random_amounts.sum()
                    random_amounts *= avg_total_precip_per_year
                    future_precip[rainy_day_indices] = random_amounts

                forecast_df[target] = future_precip
            else:
                print(f"\n  Forecasting and evaluating {target}...")
                y_train = y_train_df[target].fillna(y_train_df[target].median())
                y_test = y_test_df[target].fillna(y_test_df[target].median())

                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=5)

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)

                self.create_evaluation_plots(y_test, y_pred_test, dates_test, target)

                print(f"    Retraining on full historical data for final forecast...")
                y_historical_full = historical_df_featured[target].fillna(historical_df_featured[target].median())
                model.fit(X, y_historical_full)

                history = self.df.copy()
                future_predictions = []
                for date in future_dates:
                    temp_df = pd.concat([history, pd.DataFrame([{'date': date}])], ignore_index=True)
                    temp_featured = self.create_features(temp_df)

                    X_pred_day = temp_featured[features].iloc[-1].to_frame().T

                    prediction = model.predict(X_pred_day)[0]
                    future_predictions.append(prediction)

                    new_row = {'date': date}
                    for col in target_variables:
                        if col == target:
                            new_row[col] = prediction
                        elif col in forecast_df.columns:
                            new_row[col] = forecast_df.loc[forecast_df['date'] == date, col].iloc[0] if not \
                            forecast_df.loc[forecast_df['date'] == date].empty else 0
                        else:
                            new_row[col] = 0

                    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

                forecast_df[target] = future_predictions

        forecast_df['soil_moisture_0_to_10cm_mean'] = np.nan

        output_filename = 'future_weather_data.csv'
        forecast_df.to_csv(output_filename, index=False)
        print(f"\n✅ Forecast for the next 366 days saved to '{output_filename}'")
        print(f"   Predicted total precipitation for the year: {forecast_df['precipitation_sum'].sum():.2f}mm")


def main():
    forecaster = WeatherForecaster()
    forecaster.create_forecast()


if __name__ == "__main__":
    main()
