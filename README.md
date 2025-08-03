# Agricultural AI: Crop Yield Prediction System

*Created by [Alaa Labban](https://github.com/AlaaLabban)*

## 1. Overview

This project is a complete, end-to-end pipeline for agricultural forecasting. It leverages historical weather data to train a suite of machine learning models that can predict future crop yields. The system is designed to be adaptable to different locations and crop types, providing valuable insights for agricultural planning.

The core of the project is a two-stage forecasting process:
1.  **Weather Forecasting**: It first creates its own daily weather forecast for the next year by learning the seasonal patterns from decades of historical data.
2.  **Crop Yield Prediction**: It then uses this generated weather forecast as input for a second set of models, which predict the final yields for multiple crops under different irrigation scenarios.

---

## 2. Project Workflow

The entire process is broken down into three main steps, executed by three separate scripts.

### Step 1: Train the Crop Models
- **Script**: `model_maker_complex.py`
- **Input**: A historical daily weather data file (e.g., `jeddah_weather_clean.csv`).
- **Process**: This script reads the historical weather, creates a summarized dataset of seasonal features for each crop, and trains a `RandomForestRegressor` model for each one. It uses cross-validation to test the reliability of each model.
- **Output**: A single file, `crop_predictor.pkl`, which contains all the reliable, trained models.

### Step 2: Create the Weather Forecast
- **Script**: `create_weather_forecast.py`
- **Input**: The same historical daily weather data file.
- **Process**: This script trains its own time-series models to learn the patterns of predictable weather variables (like temperature and radiation). It then generates a day-by-day forecast for the next 366 days. For unpredictable variables like precipitation, it uses a smart heuristic to create a realistic scenario based on historical averages.
- **Output**: A single file, `future_weather_data.csv`, containing the complete weather forecast.

### Step 3: Predict Future Yields
- **Scripts**: `predict_yield.py` and `predict_yield_water.py`
- **Input**: `crop_predictor.pkl` and `future_weather_data.csv`.
- **Process**: These scripts load the trained models and the future forecast, then calculate the final yield predictions.
  - `predict_yield.py` predicts the yield based on natural rainfall.
  - `predict_yield_water.py` predicts the yield assuming perfect irrigation is provided, and calculates the amount of water needed.
- **Output**: A detailed report printed to the console.

---

## 3. How to Use

1.  **Prepare Your Data**: Ensure your historical weather data is in a CSV file (e.g., `jeddah_weather_clean.csv`). The required columns are listed in the scripts.
2.  **Train Models**: Run the training script once.
    ```bash
    python model_maker_complex.py
    ```
3.  **Create Forecast**: Run the weather forecasting script.
    ```bash
    python create_weather_forecast.py
    ```
4.  **Get Predictions**: Run either prediction script to see the results.
    ```bash
    # For natural, rain-fed conditions
    python predict_yield.py

    # For perfectly irrigated conditions
    python predict_yield_water.py
    ```

---

## 4. Customization

### Changing Location
1.  Obtain a new historical weather CSV file for the new location.
2.  In `model_maker_complex.py` and `create_weather_forecast.py`, update the filename in the `__init__` method to point to your new file.
3.  Re-run the entire workflow.

### Adding or Modifying Crops
- All crop-specific parameters (optimal temperature, water requirement, etc.) are defined in the `crop_parameters` dictionary at the top of all three main scripts.
- To add a new crop, add its definition to the dictionary in all three files.
- To modify an existing crop, edit its parameters, ensuring the changes are consistent across all three scripts.

---

## 5. Dependencies

This project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
