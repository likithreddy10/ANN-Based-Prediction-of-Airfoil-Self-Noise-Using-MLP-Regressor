"""
Airfoil Self-Noise ANN Demonstration Script
Author: JUTURU LIKITH REDDY (2023004582)
Project: ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor (SGD)
Course: Unit 3 / CO3 - Aerospace / Engineering
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

FEATURE_NAMES = [
    'frequency',
    'angle_of_attack',
    'chord_length',
    'free_stream_velocity',
    'suction_side_displacement_thickness'
]

def main():
    print("=" * 70)
    print("AIRFOIL SELF-NOISE: ANN REGRESSION DEMONSTRATION")
    print("Student: JUTURU LIKITH REDDY (2023004582)")
    print("Optimizer: Stochastic Gradient Descent (SGD)")
    print("=" * 70)

    # 1. Load Scalers and Models
    print("\nLoading trained neural models and standard scalers...")
    scaler_x = joblib.load('model_artifacts/scaler_x.joblib')
    scaler_y = joblib.load('model_artifacts/scaler_y.joblib')
    baseline_model = joblib.load('model_artifacts/baseline_model.joblib')
    proposed_model = joblib.load('model_artifacts/proposed_model.joblib')
    print("Models loaded successfully!")

    # 2. Interactive Single-Point Prediction
    sample_flight = {
        'frequency': 1000.0,
        'angle_of_attack': 5.4,
        'chord_length': 0.1524,
        'free_stream_velocity': 55.5,
        'suction_side_displacement_thickness': 0.0050
    }
    print(f"\n[SAMPLE FLIGHT PREDICTION]")
    for k, v in sample_flight.items():
        print(f"  {k:<36}: {v}")

    df_in = pd.DataFrame([sample_flight], columns=FEATURE_NAMES)
    x_s = scaler_x.transform(df_in)

    # Predict
    pred_base_s = baseline_model.predict(x_s)
    pred_base = scaler_y.inverse_transform(pred_base_s.reshape(-1, 1))[0][0]

    pred_prop_s = proposed_model.predict(x_s)
    pred_prop = scaler_y.inverse_transform(pred_prop_s.reshape(-1, 1))[0][0]

    print("\nInference Results:")
    print(f"  Baseline Model (1 Dense 32)      : {pred_base:.2f} dB")
    print(f"  Proposed Deep Model (4 Layers)   : {pred_prop:.2f} dB")
    print(f"  Architecture Delta (dB)          : {pred_prop - pred_base:+.2f} dB")

    # 3. Validation on Unseen Test Dataset
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST DATASET (airfoil_self_noise.dat)")
    print("=" * 70)
    df_raw = pd.read_csv('airfoil_self_noise.dat', sep=r'\s+', header=None)
    X = df_raw.iloc[:, :5].values
    y = df_raw.iloc[:, 5].values

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    X_test_s = scaler_x.transform(X_test_df)

    # Evaluate Proposed
    y_prop_s = proposed_model.predict(X_test_s)
    y_prop = scaler_y.inverse_transform(y_prop_s.reshape(-1, 1)).ravel()

    # Evaluate Baseline
    y_base_s = baseline_model.predict(X_test_s)
    y_base = scaler_y.inverse_transform(y_base_s.reshape(-1, 1)).ravel()

    print(f"\nBaseline ANN Test Performance (301 holdout samples):")
    print(f"  R² Score  : {r2_score(y_test, y_base):.4f}")
    print(f"  RMSE (dB) : {root_mean_squared_error(y_test, y_base):.2f} dB")
    print(f"  MAE (dB)  : {mean_absolute_error(y_test, y_base):.2f} dB")

    print(f"\nProposed Deep ANN Test Performance (301 holdout samples):")
    print(f"  R² Score  : {r2_score(y_test, y_prop):.4f}")
    print(f"  RMSE (dB) : {root_mean_squared_error(y_test, y_prop):.2f} dB")
    print(f"  MAE (dB)  : {mean_absolute_error(y_test, y_prop):.2f} dB")

    rmse_reduction = (1 - (root_mean_squared_error(y_test, y_prop) / root_mean_squared_error(y_test, y_base))) * 100
    print(f"\nError Reduction Achieved: {rmse_reduction:.1f}% reduction in RMSE!")
    print("\nTo launch the interactive web application, run:")
    print("  python app.py  (or double click start_website.bat)")
    print("Then open http://localhost:5000 in your browser.")

if __name__ == '__main__':
    main()
