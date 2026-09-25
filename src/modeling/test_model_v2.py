import os
import sys
import joblib
import pandas as pd

# 1. Dynamic Path Resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 2. File Path Configuration
    model_path = os.path.join(PROJECT_ROOT, "data", "rf_heat_v2.joblib")

    print("==========================================")
    print("     ThermoRoute - Model 2 Testing       ")
    print("==========================================")

    # 3. Verify Model File Existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}"
        )

    # 4. Load Trained Model
    print(f"Loading model from: {model_path} ...")
    loaded_model = joblib.load(model_path)
    print(" SUCCESS: Model loaded into memory successfully!\n")

    # 5. Prepare Test Features (Matching training columns: length & speed)
    test_streets = pd.DataFrame({
        'length': [100.0, 250.0, 500.0, 1000.0],
        'speed': [30.0, 40.0, 20.0, 60.0]
    })

    # 6. Generate Predictions
    predictions = loaded_model.predict(test_streets)

    # 7. Display Results
    print("--- Heat Index Prediction Results ---")
    for i, pred in enumerate(predictions):
        length = test_streets.loc[i, 'length']
        speed = test_streets.loc[i, 'speed']
        print(f"Street {i+1} (Length: {length}m, Speed: {speed}km/h) -> Predicted Heat Index: {pred:.2f}")

    print("==========================================")


if __name__ == "__main__":
    main()