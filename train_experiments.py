import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure writable matplotlib temp config dir
os.environ['MPLCONFIGDIR'] = os.path.abspath('static/charts')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature and target definitions
FEATURE_NAMES = [
    'frequency',
    'angle_of_attack',
    'chord_length',
    'free_stream_velocity',
    'suction_side_displacement_thickness'
]
FEATURE_LABELS = {
    'frequency': 'Frequency (Hz)',
    'angle_of_attack': 'Angle of Attack (deg)',
    'chord_length': 'Chord Length (m)',
    'free_stream_velocity': 'Free-stream Velocity (m/s)',
    'suction_side_displacement_thickness': 'Suction Side Displacement Thickness (m)'
}
FEATURE_UNITS = {
    'frequency': 'Hz',
    'angle_of_attack': '° (deg)',
    'chord_length': 'm',
    'free_stream_velocity': 'm/s',
    'suction_side_displacement_thickness': 'm'
}
TARGET_NAME = 'sound_pressure_level'
TARGET_LABEL = 'Scaled Sound Pressure Level (dB)'

def load_and_preprocess_data(filepath='airfoil_self_noise.dat'):
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=FEATURE_NAMES + [TARGET_NAME])
    
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    y_train_arr = y_train.values.reshape(-1, 1)
    y_test_arr = y_test.values.reshape(-1, 1)
    
    y_train_scaled = scaler_y.fit_transform(y_train_arr).ravel()
    y_test_scaled = scaler_y.transform(y_test_arr).ravel()
    
    # Dataset statistics
    feature_stats = {}
    for col in FEATURE_NAMES + [TARGET_NAME]:
        s = df[col]
        feature_stats[col] = {
            'min': float(s.min()),
            'max': float(s.max()),
            'mean': float(s.mean()),
            'std': float(s.std()),
            'median': float(s.median()),
            'q25': float(s.quantile(0.25)),
            'q75': float(s.quantile(0.75))
        }
        
    summary = {
        'num_samples': len(df),
        'num_features': len(FEATURE_NAMES),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_stats': feature_stats,
        'feature_names': FEATURE_NAMES,
        'feature_labels': FEATURE_LABELS,
        'feature_units': FEATURE_UNITS,
        'target_name': TARGET_NAME,
        'target_label': TARGET_LABEL,
        'train_indices': X_train.index.tolist(),
        'test_indices': X_test.index.tolist()
    }
    
    return df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y, summary

def calc_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def evaluate_scaled_model(model, scaler_y, X_train_s, y_train, X_test_s, y_test):
    # Predict in scaled domain then transform back to dB
    y_train_pred_s = model.predict(X_train_s)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_s.reshape(-1, 1)).ravel()
    
    y_test_pred_s = model.predict(X_test_s)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()
    
    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)
    
    residuals = (y_test - y_test_pred).tolist()
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'test_predictions': [round(p, 4) for p in y_test_pred.tolist()],
        'test_actuals': [round(a, 4) for a in y_test.tolist()],
        'residuals': [round(r, 4) for r in residuals]
    }

def run_all_experiments():
    os.makedirs('static/charts', exist_ok=True)
    os.makedirs('model_artifacts', exist_ok=True)
    
    print('='*70)
    print('AIRFOIL SELF-NOISE: ANN REGRESSION EXPERIMENTAL BENCHMARK')
    print('Student: JUTURU LIKITH REDDY (Roll No: 2023004582)')
    print('Domain: Aerospace / Engineering | Course: Unit 3 / CO3')
    print('Optimizer: Stochastic Gradient Descent (SGD)')
    print('='*70)
    
    df, X_train, X_test, y_train, y_test, X_train_s, X_test_s, y_train_s, y_test_s, scaler_x, scaler_y, summary = load_and_preprocess_data()
    
    # Save scalers
    joblib.dump(scaler_x, 'model_artifacts/scaler_x.joblib')
    joblib.dump(scaler_y, 'model_artifacts/scaler_y.joblib')
    
    all_results = {
        'student_info': {
            'name': 'JUTURU LIKITH REDDY',
            'roll_no': '2023004582',
            'domain': 'Aerospace / Engineering',
            'project_title': 'ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor',
            'unit_co': 'Unit 3 / CO3',
            'dataset_source': 'UCI Machine Learning Repository (ID 291)',
            'dataset_url': 'https://archive.ics.uci.edu/dataset/291/airfoil+self-noise',
            'observations': 1503,
            'features_count': 5,
            'target': 'Scaled Sound Pressure Level (dB)',
            'optimizer': 'Stochastic Gradient Descent (SGD)',
            'random_seed': RANDOM_SEED,
            'train_test_split': '80% Train (1202 samples), 20% Test (301 samples)'
        },
        'summary': summary,
        'experiments': {}
    }
    
    # -------------------------------------------------------------
    # 1. BASELINE MODEL
    # -------------------------------------------------------------
    print('\n[1/5] Training Baseline ANN (Single Dense 32, Constant LR SGD)...')
    t0 = time.time()
    baseline_mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='sgd',
        learning_rate='constant',
        learning_rate_init=0.01,
        momentum=0.0,
        batch_size=64,
        max_iter=600,
        random_state=RANDOM_SEED
    )
    baseline_mlp.fit(X_train_s, y_train_s)
    t_baseline = time.time() - t0
    
    baseline_eval = evaluate_scaled_model(baseline_mlp, scaler_y, X_train_s, y_train, X_test_s, y_test)
    all_results['experiments']['baseline'] = {
        'name': 'Baseline ANN',
        'architecture_desc': 'Input(5) -> Dense(32, ReLU) -> Output(1)',
        'hidden_layer_sizes': [32],
        'solver': 'sgd (standard constant lr)',
        'learning_rate': 'constant (0.01)',
        'batch_size': 64,
        'momentum': 0.0,
        'epochs_converged': int(baseline_mlp.n_iter_),
        'train_time_sec': round(t_baseline, 4),
        'metrics': baseline_eval,
        'loss_curve': [float(l) for l in baseline_mlp.loss_curve_]
    }
    joblib.dump(baseline_mlp, 'model_artifacts/baseline_model.joblib')
    print(f"  Baseline -> Test R2: {baseline_eval['test']['r2']:.4f} | RMSE: {baseline_eval['test']['rmse']:.4f} dB | MAE: {baseline_eval['test']['mae']:.4f} dB | Iter: {baseline_mlp.n_iter_}")
    
    # -------------------------------------------------------------
    # 2. PROPOSED ARCHITECTURE
    # -------------------------------------------------------------
    print('\n[2/5] Training Proposed Architecture (Deep MLP 256-128-64-32, SGD + Nesterov Momentum + Adaptive LR)...')
    t0 = time.time()
    proposed_mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='sgd',
        learning_rate='adaptive',
        learning_rate_init=0.01,
        momentum=0.9,
        nesterovs_momentum=True,
        batch_size=32,
        alpha=0.0001,
        max_iter=800,
        random_state=RANDOM_SEED
    )
    proposed_mlp.fit(X_train_s, y_train_s)
    t_proposed = time.time() - t0
    
    proposed_eval = evaluate_scaled_model(proposed_mlp, scaler_y, X_train_s, y_train, X_test_s, y_test)
    all_results['experiments']['proposed'] = {
        'name': 'Proposed Deep ANN Architecture',
        'architecture_desc': 'Input(5) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1)',
        'hidden_layer_sizes': [256, 128, 64, 32],
        'solver': 'sgd (Nesterov Momentum 0.9 + Adaptive LR)',
        'learning_rate': 'adaptive (initial 0.01)',
        'batch_size': 32,
        'momentum': 0.9,
        'epochs_converged': int(proposed_mlp.n_iter_),
        'train_time_sec': round(t_proposed, 4),
        'metrics': proposed_eval,
        'loss_curve': [float(l) for l in proposed_mlp.loss_curve_]
    }
    joblib.dump(proposed_mlp, 'model_artifacts/proposed_model.joblib')
    print(f"  Proposed -> Test R2: {proposed_eval['test']['r2']:.4f} | RMSE: {proposed_eval['test']['rmse']:.4f} dB | MAE: {proposed_eval['test']['mae']:.4f} dB | Iter: {proposed_mlp.n_iter_}")

    # -------------------------------------------------------------
    # 3. CONTROLLED VARIATION 1: Learning Rate
    # -------------------------------------------------------------
    print('\n[3/5] Controlled Experiment 1: Learning Rate Variation [0.001, 0.005, 0.01, 0.02, 0.05]...')
    lr_candidates = [0.001, 0.005, 0.01, 0.02, 0.05]
    lr_results = []
    
    for lr in lr_candidates:
        t_start = time.time()
        m = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            learning_rate_init=lr,
            momentum=0.9,
            batch_size=32,
            max_iter=600,
            random_state=RANDOM_SEED
        )
        m.fit(X_train_s, y_train_s)
        elapsed = time.time() - t_start
        eval_res = evaluate_scaled_model(m, scaler_y, X_train_s, y_train, X_test_s, y_test)
        
        lr_results.append({
            'learning_rate': lr,
            'epochs_converged': int(m.n_iter_),
            'train_time_sec': round(elapsed, 4),
            'train_rmse': round(eval_res['train']['rmse'], 4),
            'train_r2': round(eval_res['train']['r2'], 4),
            'test_mae': round(eval_res['test']['mae'], 4),
            'test_mse': round(eval_res['test']['mse'], 4),
            'test_rmse': round(eval_res['test']['rmse'], 4),
            'test_r2': round(eval_res['test']['r2'], 4),
            'loss_curve': [float(l) for l in m.loss_curve_]
        })
        print(f"  LR={lr:<5} | Test R2: {eval_res['test']['r2']:.4f} | Test RMSE: {eval_res['test']['rmse']:.4f} dB | Epochs: {m.n_iter_}")
    all_results['experiments']['learning_rate_variation'] = lr_results

    # -------------------------------------------------------------
    # 4. CONTROLLED VARIATION 2: Batch Size
    # -------------------------------------------------------------
    print('\n[4/5] Controlled Experiment 2: Batch Size Variation [16, 32, 64, 128, 256]...')
    batch_candidates = [16, 32, 64, 128, 256]
    batch_results = []
    
    for bs in batch_candidates:
        t_start = time.time()
        m = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            learning_rate_init=0.01,
            momentum=0.9,
            batch_size=bs,
            max_iter=600,
            random_state=RANDOM_SEED
        )
        m.fit(X_train_s, y_train_s)
        elapsed = time.time() - t_start
        eval_res = evaluate_scaled_model(m, scaler_y, X_train_s, y_train, X_test_s, y_test)
        
        batch_results.append({
            'batch_size': bs,
            'epochs_converged': int(m.n_iter_),
            'train_time_sec': round(elapsed, 4),
            'train_rmse': round(eval_res['train']['rmse'], 4),
            'train_r2': round(eval_res['train']['r2'], 4),
            'test_mae': round(eval_res['test']['mae'], 4),
            'test_mse': round(eval_res['test']['mse'], 4),
            'test_rmse': round(eval_res['test']['rmse'], 4),
            'test_r2': round(eval_res['test']['r2'], 4),
            'loss_curve': [float(l) for l in m.loss_curve_]
        })
        print(f"  Batch={bs:<4} | Test R2: {eval_res['test']['r2']:.4f} | Test RMSE: {eval_res['test']['rmse']:.4f} dB | Time: {elapsed:.2f}s")
    all_results['experiments']['batch_size_variation'] = batch_results

    # -------------------------------------------------------------
    # 5. CONTROLLED VARIATION 3: Architecture & Activation
    # -------------------------------------------------------------
    print('\n[5/5] Controlled Experiment 3: Architecture Depth & Activation Functions...')
    arch_configs = [
        {'name': 'Shallow (32)', 'layers': (32,), 'act': 'relu'},
        {'name': 'Two-Layer (64, 32)', 'layers': (64, 32), 'act': 'relu'},
        {'name': 'Three-Layer (128, 64, 32)', 'layers': (128, 64, 32), 'act': 'relu'},
        {'name': 'Deep Three-Layer (128, 128, 64)', 'layers': (128, 128, 64), 'act': 'relu'},
        {'name': 'Proposed Four-Layer (256, 128, 64, 32)', 'layers': (256, 128, 64, 32), 'act': 'relu'},
        {'name': 'Tanh Activation (128, 64, 32)', 'layers': (128, 64, 32), 'act': 'tanh'},
        {'name': 'Logistic Activation (128, 64, 32)', 'layers': (128, 64, 32), 'act': 'logistic'}
    ]
    arch_results = []
    
    for cfg in arch_configs:
        t_start = time.time()
        m = MLPRegressor(
            hidden_layer_sizes=cfg['layers'],
            activation=cfg['act'],
            solver='sgd',
            learning_rate='adaptive',
            learning_rate_init=0.01,
            momentum=0.9,
            batch_size=32,
            max_iter=600,
            random_state=RANDOM_SEED
        )
        m.fit(X_train_s, y_train_s)
        elapsed = time.time() - t_start
        eval_res = evaluate_scaled_model(m, scaler_y, X_train_s, y_train, X_test_s, y_test)
        
        arch_results.append({
            'config_name': cfg['name'],
            'hidden_layers': list(cfg['layers']),
            'activation': cfg['act'],
            'epochs_converged': int(m.n_iter_),
            'train_time_sec': round(elapsed, 4),
            'train_rmse': round(eval_res['train']['rmse'], 4),
            'train_r2': round(eval_res['train']['r2'], 4),
            'test_mae': round(eval_res['test']['mae'], 4),
            'test_mse': round(eval_res['test']['mse'], 4),
            'test_rmse': round(eval_res['test']['rmse'], 4),
            'test_r2': round(eval_res['test']['r2'], 4)
        })
        print(f"  {cfg['name']:<35} | Test R2: {eval_res['test']['r2']:.4f} | Test RMSE: {eval_res['test']['rmse']:.4f} dB")
    all_results['experiments']['architecture_variation'] = arch_results

    # Save JSON results
    with open('model_artifacts/experiments_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\n[SUCCESS] Experimental evaluation complete. Results written to model_artifacts/experiments_results.json')
    
    # Generate high quality charts for report & website
    generate_charts(df, all_results, proposed_mlp, baseline_mlp, scaler_y, X_test_s, y_test)
    print('[SUCCESS] Visual comparison plots generated in static/charts/')

def generate_charts(df, results, proposed_model, baseline_model, scaler_y, X_test_s, y_test):
    # Set high DPI and aesthetic styling
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # 1. Actual vs Predicted Scatter (Baseline vs Proposed)
    y_pred_proposed_s = proposed_model.predict(X_test_s)
    y_pred_proposed = scaler_y.inverse_transform(y_pred_proposed_s.reshape(-1, 1)).ravel()
    
    y_pred_baseline_s = baseline_model.predict(X_test_s)
    y_pred_baseline = scaler_y.inverse_transform(y_pred_baseline_s.reshape(-1, 1)).ravel()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline subplot
    axes[0].scatter(y_test, y_pred_baseline, alpha=0.5, color='#ef4444', s=35, edgecolors='none')
    min_v, max_v = y_test.min(), y_test.max()
    axes[0].plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal y = x')
    axes[0].set_title(f"Baseline ANN (R²={results['experiments']['baseline']['metrics']['test']['r2']:.4f})", fontweight='bold')
    axes[0].set_xlabel('Actual Sound Pressure Level (dB)')
    axes[0].set_ylabel('Predicted Sound Pressure Level (dB)')
    axes[0].legend()
    
    # Proposed subplot
    axes[1].scatter(y_test, y_pred_proposed, alpha=0.6, color='#2563eb', s=35, edgecolors='none')
    axes[1].plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal y = x')
    axes[1].set_title(f"Proposed Deep ANN (R²={results['experiments']['proposed']['metrics']['test']['r2']:.4f})", fontweight='bold')
    axes[1].set_xlabel('Actual Sound Pressure Level (dB)')
    axes[1].set_ylabel('Predicted Sound Pressure Level (dB)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('static/charts/actual_vs_predicted.png', dpi=200)
    plt.close()

    # 2. Residual Distribution Plot
    res_prop = y_test.values - y_pred_proposed
    res_base = y_test.values - y_pred_baseline
    
    plt.figure(figsize=(9, 5))
    sns.kdeplot(res_base, color='#ef4444', label=f'Baseline (Std Error={np.std(res_base):.2f} dB)', lw=2)
    sns.kdeplot(res_prop, color='#2563eb', label=f'Proposed (Std Error={np.std(res_prop):.2f} dB)', lw=2.5, fill=True, alpha=0.2)
    plt.axvline(0, color='black', linestyle='--', lw=1.2)
    plt.title('Residual Error Distribution (Actual - Predicted dB)', fontweight='bold')
    plt.xlabel('Error (dB)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/charts/residual_distribution.png', dpi=200)
    plt.close()

    # 3. Loss Curves: Baseline vs Proposed
    plt.figure(figsize=(9, 5))
    plt.plot(results['experiments']['baseline']['loss_curve'], label='Baseline ANN (Constant LR 0.01)', color='#ef4444', lw=2)
    plt.plot(results['experiments']['proposed']['loss_curve'], label='Proposed ANN (Adaptive LR + Nesterov 0.9)', color='#2563eb', lw=2)
    plt.title('Training Loss Convergence per Epoch (SGD)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Squared Error in Scaled Domain)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/charts/loss_curves_comparison.png', dpi=200)
    plt.close()

    # 4. Correlation Heatmap of Dataset
    plt.figure(figsize=(9, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.8, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Airfoil Self-Noise Aerodynamic Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/correlation_heatmap.png', dpi=200)
    plt.close()

    # 5. Controlled Variations Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LR variation
    lrs = [x['learning_rate'] for x in results['experiments']['learning_rate_variation']]
    r2_lrs = [x['test_r2'] for x in results['experiments']['learning_rate_variation']]
    axes[0].plot([str(lr) for lr in lrs], r2_lrs, marker='o', lw=2.5, color='#0284c7', markersize=8)
    axes[0].set_title('Controlled Experiment 1: Test R² vs. Learning Rate', fontweight='bold')
    axes[0].set_xlabel('SGD Learning Rate')
    axes[0].set_ylabel('Test R² Score')
    axes[0].set_ylim(0.7, 1.0)
    for i, txt in enumerate(r2_lrs):
        axes[0].annotate(f"{txt:.3f}", (str(lrs[i]), r2_lrs[i] + 0.01), ha='center', fontsize=9, fontweight='bold')
        
    # Batch size variation
    bs_vals = [x['batch_size'] for x in results['experiments']['batch_size_variation']]
    r2_bs = [x['test_r2'] for x in results['experiments']['batch_size_variation']]
    axes[1].plot([str(b) for b in bs_vals], r2_bs, marker='s', lw=2.5, color='#10b981', markersize=8)
    axes[1].set_title('Controlled Experiment 2: Test R² vs. Batch Size', fontweight='bold')
    axes[1].set_xlabel('Mini-Batch Size')
    axes[1].set_ylabel('Test R² Score')
    axes[1].set_ylim(0.7, 1.0)
    for i, txt in enumerate(r2_bs):
        axes[1].annotate(f"{txt:.3f}", (str(bs_vals[i]), r2_bs[i] + 0.01), ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('static/charts/controlled_variations.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    run_all_experiments()
