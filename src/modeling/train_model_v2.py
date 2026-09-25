"""
ThermoRoute - Model 2 Training Script
Trains the Alternative Heat Exposure Model (Random Forest / LightGBM)
and saves the serialized model (.joblib) for production use.
"""

import os
import pandas as pd
from src.modeling.heat_exposure_model_v2 import AlternativeHeatExposureModel


def main():
    # File Paths Configuration
    data_path = "data/processed_features.csv"
    output_model_path = "data/rf_heat_v2.joblib"
    target_column = "heat_index"

    print("==========================================")
    print("      ThermoRoute - Training Model 2      ")
    print("==========================================")

    # 1. Check if dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please verify the path."
        )

    # 2. Load Processed Dataset
    print(f"[1/4] Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
        )

    # 3. Separate Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"      Dataset successfully loaded. Features shape: {X.shape}, Target shape: {y.shape}")

    # 4. Initialize Alternative Model (Model 2)
    print("[2/4] Initializing Model 2 (Random Forest Regressor)...")
    model_v2 = AlternativeHeatExposureModel(
        model_type="rf",
        n_estimators=100,
        max_depth=12,
        random_state=42
    )

    # 5. Train the Model
    print("[3/4] Training Model 2 on dataset...")
    model_v2.train(X, y)

    # 6. Save the Trained Model
    print(f"[4/4] Saving trained model weights to: {output_model_path}...")
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model_v2.save_model(output_model_path)

    print("==========================================")
    print(" SUCCESS: Model 2 training completed!")
    print(f" Trained model file saved as: {output_model_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
