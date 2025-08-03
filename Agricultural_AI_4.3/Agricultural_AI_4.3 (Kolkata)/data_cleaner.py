import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


class WeatherDataDiagnostics:
    def __init__(self):
        pass

    def load_latest_data(self):
        """
        Load the most recent weather data file
        """
        csv_files = glob.glob('jeddah_weather_daily_8589days*.csv')

        if not csv_files:
            print("No weather data files found!")
            return None

        latest_file = max(csv_files)
        print(f"Loading data from: {latest_file}")

        try:
            df = pd.read_csv(latest_file)
            print(f"Successfully loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def comprehensive_data_analysis(self, df):
        """
        Perform comprehensive analysis of the weather data
        """
        print("=" * 80)
        print("COMPREHENSIVE WEATHER DATA ANALYSIS")
        print("=" * 80)

        # Basic info
        print(f"\n📊 BASIC INFORMATION:")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Total days: {(df['date'].max() - df['date'].min()).days}")

        # Data types analysis
        print(f"\n🔍 DATA TYPES:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")

        # Identify problematic columns
        print(f"\n⚠️  POTENTIAL ISSUES:")

        # Check for object columns that should be numeric
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 1:  # More than just 'date'
            print(f"   Object columns (should be numeric?): {list(object_cols)}")

            for col in object_cols:
                if col != 'date':
                    unique_vals = df[col].unique()[:10]  # Show first 10 unique values
                    print(f"     {col}: {unique_vals}")

        # Check for missing values
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            print(f"\n❌ MISSING VALUES:")
            for col, missing_count in missing_data.head(10).items():
                percentage = (missing_count / len(df)) * 100
                print(f"   {col}: {missing_count} ({percentage:.1f}%)")
        else:
            print(f"\n✅ NO MISSING VALUES")

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n📈 NUMERIC COLUMNS ({len(numeric_cols)}):")

        if len(numeric_cols) > 0:
            # Basic statistics
            stats = df[numeric_cols].describe()
            print(
                f"   Temperature range: {stats.loc['min', 'temperature_2m_max']:.1f}°C to {stats.loc['max', 'temperature_2m_max']:.1f}°C")

            if 'precipitation_sum' in numeric_cols:
                print(
                    f"   Precipitation range: {stats.loc['min', 'precipitation_sum']:.1f}mm to {stats.loc['max', 'precipitation_sum']:.1f}mm")

            # Check for outliers
            print(f"\n🎯 OUTLIER DETECTION:")
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"   {col}: {len(outliers)} outliers ({len(outliers) / len(df) * 100:.1f}%)")

        return df

    def fix_data_types(self, df):
        """
        Fix common data type issues
        """
        print(f"\n🔧 FIXING DATA TYPES...")

        df_fixed = df.copy()

        # Convert date column
        if 'date' in df_fixed.columns:
            df_fixed['date'] = pd.to_datetime(df_fixed['date'])

        # Fix numeric columns that might be stored as objects
        for col in df_fixed.columns:
            if col != 'date' and df_fixed[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                    print(f"   ✓ Converted {col} to numeric")
                except:
                    print(f"   ✗ Could not convert {col} to numeric")

        # Remove or fix categorical columns
        categorical_cols = ['season']
        for col in categorical_cols:
            if col in df_fixed.columns:
                if df_fixed[col].dtype == 'object':
                    # Convert to numeric encoding
                    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                    df_fixed[col + '_encoded'] = df_fixed[col].map(season_map)
                    df_fixed = df_fixed.drop(col, axis=1)
                    print(f"   ✓ Encoded {col} as numeric")

        return df_fixed

    def create_diagnostic_plots(self, df):
        """
        Create diagnostic plots
        """
        print(f"\n📊 CREATING DIAGNOSTIC PLOTS...")

        # Select key variables for plotting
        key_vars = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
        available_vars = [var for var in key_vars if var in df.columns]

        if len(available_vars) == 0:
            print("   No key variables available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weather Data Diagnostics', fontsize=16, fontweight='bold')

        # Plot 1: Time series of temperature
        if 'temperature_2m_max' in available_vars and 'date' in df.columns:
            axes[0, 0].plot(df['date'], df['temperature_2m_max'])
            axes[0, 0].set_title('Temperature Over Time')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Precipitation distribution
        if 'precipitation_sum' in available_vars:
            axes[0, 1].hist(df['precipitation_sum'], bins=50, alpha=0.7)
            axes[0, 1].set_title('Precipitation Distribution')
            axes[0, 1].set_xlabel('Precipitation (mm)')
            axes[0, 1].set_ylabel('Frequency')

        # Plot 3: Missing data heatmap
        missing_data = df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data.iloc[:, :20], cbar=True, ax=axes[1, 0])  # First 20 columns
            axes[1, 0].set_title('Missing Data Pattern')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Missing Data Pattern')

        # Plot 4: Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # First 10 numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Correlation Matrix')

        plt.tight_layout()
        plt.show()

    def prepare_clean_dataset(self, df):
        """
        Prepare a clean dataset for prediction
        """
        print(f"\n🧹 PREPARING CLEAN DATASET...")

        # Fix data types
        df_clean = self.fix_data_types(df)

        # Remove columns with too many missing values (>50%)
        missing_threshold = 0.5
        cols_to_keep = []

        for col in df_clean.columns:
            if col == 'date':
                cols_to_keep.append(col)
            else:
                missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                if missing_pct <= missing_threshold:
                    cols_to_keep.append(col)
                else:
                    print(f"   ✗ Removing {col} (missing: {missing_pct * 100:.1f}%)")

        df_clean = df_clean[cols_to_keep]
        print(f"   ✓ Kept {len(cols_to_keep)} columns")

        # Save clean dataset
        clean_filename = 'jeddah_weather_clean.csv'
        df_clean.to_csv(clean_filename, index=False)
        print(f"   ✓ Saved clean dataset: {clean_filename}")

        return df_clean


def main():
    """
    Main diagnostic function
    """
    diagnostics = WeatherDataDiagnostics()

    # Load data
    df = diagnostics.load_latest_data()
    if df is None:
        return

    # Comprehensive analysis
    df = diagnostics.comprehensive_data_analysis(df)

    # Create diagnostic plots
    diagnostics.create_diagnostic_plots(df)

    # Prepare clean dataset
    df_clean = diagnostics.prepare_clean_dataset(df)

    print(f"\n✅ DIAGNOSTICS COMPLETE!")
    print(f"   Clean dataset ready for prediction: jeddah_weather_clean.csv")
    print(f"   Shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
