import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
import io

app = Flask(__name__)

# Constants & Physics
SPEED_OF_SOUND = 343.0  # m/s at 20°C in air
KINEMATIC_VISCOSITY_AIR = 1.5e-5  # m^2/s

# Feature metadata
FEATURE_KEYS = [
    'frequency',
    'angle_of_attack',
    'chord_length',
    'free_stream_velocity',
    'suction_side_displacement_thickness'
]

FEATURE_META = {
    'frequency': {
        'name': 'Frequency',
        'unit': 'Hz',
        'min': 200.0,
        'max': 20000.0,
        'default': 1000.0,
        'step': 50.0,
        'desc': 'Acoustic frequency in Hertz of the self-noise radiation.'
    },
    'angle_of_attack': {
        'name': 'Angle of Attack (α)',
        'unit': 'degrees (°)',
        'min': 0.0,
        'max': 22.2,
        'default': 5.4,
        'step': 0.1,
        'desc': 'Airfoil geometric angle relative to incoming freestream airflow.'
    },
    'chord_length': {
        'name': 'Chord Length (c)',
        'unit': 'meters (m)',
        'min': 0.0254,
        'max': 0.3048,
        'default': 0.1524,
        'step': 0.0001,
        'desc': 'Distance between the leading edge and trailing edge of the airfoil blade.'
    },
    'free_stream_velocity': {
        'name': 'Free-stream Velocity (U∞)',
        'unit': 'm/s',
        'min': 31.7,
        'max': 71.3,
        'default': 55.5,
        'step': 0.5,
        'desc': 'Speed of undisturbed air in the wind tunnel test section.'
    },
    'suction_side_displacement_thickness': {
        'name': 'Suction Side Displacement Thickness (δ*)',
        'unit': 'meters (m)',
        'min': 0.000401,
        'max': 0.058411,
        'default': 0.005,
        'step': 0.0001,
        'desc': 'Boundary layer displacement thickness measured at the airfoil trailing edge.'
    }
}

# Load Models and Scalers
try:
    scaler_x = joblib.load('model_artifacts/scaler_x.joblib')
    scaler_y = joblib.load('model_artifacts/scaler_y.joblib')
    baseline_model = joblib.load('model_artifacts/baseline_model.joblib')
    proposed_model = joblib.load('model_artifacts/proposed_model.joblib')
except Exception as e:
    print(f"Error loading models or scalers: {e}")
    scaler_x = None
    scaler_y = None
    baseline_model = None
    proposed_model = None

# Load Experimental Results
EXPERIMENTS_FILE = 'model_artifacts/experiments_results.json'
experiments_data = {}
if os.path.exists(EXPERIMENTS_FILE):
    with open(EXPERIMENTS_FILE, 'r') as f:
        experiments_data = json.load(f)

def compute_aero_physics(freq, aoa, chord, velocity, thickness):
    """Computes aerodynamic and acoustic non-dimensional numbers."""
    # Strouhal number based on displacement thickness: St = f * delta* / U_inf
    strouhal_delta = (freq * thickness) / velocity if velocity > 0 else 0
    
    # Chord-based Reynolds number: Re_c = U_inf * c / nu
    reynolds_chord = (velocity * chord) / KINEMATIC_VISCOSITY_AIR
    
    # Acoustic wavelength in air: lambda = c_sound / f
    wavelength = SPEED_OF_SOUND / freq if freq > 0 else 0
    
    # Noise classification
    # Typical SPL range in dataset is 103 dB to 141 dB
    return {
        'strouhal_delta': round(strouhal_delta, 4),
        'reynolds_chord': int(reynolds_chord),
        'wavelength_meters': round(wavelength, 4),
        'mach_number': round(velocity / SPEED_OF_SOUND, 3)
    }

def get_noise_category(spl):
    if spl < 115.0:
        return {'label': 'Low Noise', 'color': 'text-emerald-500', 'bg': 'bg-emerald-50 border-emerald-200', 'desc': 'Below average aeroacoustic radiation.'}
    elif spl < 125.0:
        return {'label': 'Moderate Noise', 'color': 'text-blue-500', 'bg': 'bg-blue-50 border-blue-200', 'desc': 'Nominal boundary-layer turbulence sound level.'}
    elif spl < 133.0:
        return {'label': 'Elevated Noise', 'color': 'text-amber-500', 'bg': 'bg-amber-50 border-amber-200', 'desc': 'High trailing-edge noise or separation vortex shedding.'}
    else:
        return {'label': 'Severe Acoustic Radiation', 'color': 'text-rose-500', 'bg': 'bg-rose-50 border-rose-200', 'desc': 'High-intensity aeroacoustic noise radiation near stall conditions.'}

@app.route('/')
def index():
    return render_template(
        'index.html',
        meta=FEATURE_META,
        experiments=experiments_data
    )

@app.route('/experiments')
def experiments():
    return render_template(
        'experiments.html',
        experiments=experiments_data
    )

@app.route('/dataset')
def dataset():
    # Load first 25 rows for table preview
    try:
        df_preview = pd.read_csv('airfoil_self_noise.dat', sep=r'\s+', header=None, nrows=25)
        df_preview.columns = [
            'Frequency (Hz)',
            'Angle of Attack (°)',
            'Chord Length (m)',
            'Free-stream Velocity (m/s)',
            'Displacement Thickness (m)',
            'Sound Pressure Level (dB)'
        ]
        sample_table = df_preview.to_dict(orient='records')
    except Exception:
        sample_table = []
        
    return render_template(
        'dataset.html',
        meta=FEATURE_META,
        experiments=experiments_data,
        sample_table=sample_table
    )

@app.route('/architecture')
def architecture():
    return render_template(
        'architecture.html',
        experiments=experiments_data
    )

@app.route('/report')
def report():
    return render_template(
        'report.html',
        experiments=experiments_data
    )

@app.route('/batch')
def batch():
    return render_template(
        'batch.html',
        meta=FEATURE_META
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        freq = float(data.get('frequency', 1000.0))
        aoa = float(data.get('angle_of_attack', 0.0))
        chord = float(data.get('chord_length', 0.3048))
        vel = float(data.get('free_stream_velocity', 71.3))
        thick = float(data.get('suction_side_displacement_thickness', 0.002663))

        raw_df = pd.DataFrame([[freq, aoa, chord, vel, thick]], columns=FEATURE_KEYS)
        scaled_vec = scaler_x.transform(raw_df)

        # Proposed model prediction
        pred_prop_s = proposed_model.predict(scaled_vec)
        pred_proposed = float(scaler_y.inverse_transform(pred_prop_s.reshape(-1, 1))[0][0])

        # Baseline model prediction
        pred_base_s = baseline_model.predict(scaled_vec)
        pred_baseline = float(scaler_y.inverse_transform(pred_base_s.reshape(-1, 1))[0][0])

        physics = compute_aero_physics(freq, aoa, chord, vel, thick)
        category = get_noise_category(pred_proposed)

        return jsonify({
            'success': True,
            'predicted_sound_pressure_level': round(pred_proposed, 2),
            'baseline_sound_pressure_level': round(pred_baseline, 2),
            'delta_dB': round(pred_proposed - pred_baseline, 2),
            'noise_category': category,
            'physics': physics,
            'input_values': {
                'frequency': freq,
                'angle_of_attack': aoa,
                'chord_length': chord,
                'free_stream_velocity': vel,
                'suction_side_displacement_thickness': thick
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/frequency_sweep', methods=['POST'])
def api_frequency_sweep():
    """Generates an aeroacoustic spectrum curve across frequencies for the given configuration."""
    try:
        data = request.get_json(force=True)
        aoa = float(data.get('angle_of_attack', 5.4))
        chord = float(data.get('chord_length', 0.1524))
        vel = float(data.get('free_stream_velocity', 55.5))
        thick = float(data.get('suction_side_displacement_thickness', 0.005))

        # Sweep standard 1/3-octave frequencies from 200 Hz to 20,000 Hz
        frequencies = [
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        ]
        
        raw_mat = [[f, aoa, chord, vel, thick] for f in frequencies]
        raw_df = pd.DataFrame(raw_mat, columns=FEATURE_KEYS)
        scaled_mat = scaler_x.transform(raw_df)
        
        pred_prop_s = proposed_model.predict(scaled_mat)
        preds_prop = scaler_y.inverse_transform(pred_prop_s.reshape(-1, 1)).ravel().round(2).tolist()
        
        pred_base_s = baseline_model.predict(scaled_mat)
        preds_base = scaler_y.inverse_transform(pred_base_s.reshape(-1, 1)).ravel().round(2).tolist()

        return jsonify({
            'success': True,
            'frequencies': frequencies,
            'proposed_spl': preds_prop,
            'baseline_spl': preds_base
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/experiments_data')
def api_experiments_data():
    return jsonify(experiments_data)

@app.route('/sample_csv')
def sample_csv():
    """Download a ready-to-test CSV with sample airfoil observations."""
    df = pd.read_csv('airfoil_self_noise.dat', sep=r'\s+', header=None, nrows=20)
    df.columns = [
        'frequency',
        'angle_of_attack',
        'chord_length',
        'free_stream_velocity',
        'suction_side_displacement_thickness',
        'actual_sound_pressure_level'
    ]
    csv_str = df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sample_airfoil_test_data.csv"}
    )

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        # Try reading CSV
        df = pd.read_csv(file)
        
        # Check if columns match or if index-based
        if len(df.columns) < 5:
            return jsonify({'success': False, 'error': 'CSV must have at least 5 columns for the features'}), 400
        
        # Feature columns
        X_batch_df = pd.DataFrame(df.iloc[:, :5].values, columns=FEATURE_KEYS)
        X_batch_s = scaler_x.transform(X_batch_df)
        
        # Predict
        prop_s = proposed_model.predict(X_batch_s)
        preds_prop = scaler_y.inverse_transform(prop_s.reshape(-1, 1)).ravel()
        
        base_s = baseline_model.predict(X_batch_s)
        preds_base = scaler_y.inverse_transform(base_s.reshape(-1, 1)).ravel()
        
        df_out = df.copy()
        df_out['predicted_proposed_dB'] = preds_prop.round(2)
        df_out['predicted_baseline_dB'] = preds_base.round(2)
        
        has_actual = False
        mae_prop = None
        rmse_prop = None
        r2_prop = None
        
        if len(df.columns) >= 6:
            # Assume 6th column is actual
            y_actual = df.iloc[:, 5].values
            df_out['residual_proposed_dB'] = (y_actual - preds_prop).round(2)
            has_actual = True
            from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
            mae_prop = round(float(mean_absolute_error(y_actual, preds_prop)), 4)
            rmse_prop = round(float(root_mean_squared_error(y_actual, preds_prop)), 4)
            r2_prop = round(float(r2_score(y_actual, preds_prop)), 4)

        results_list = df_out.head(100).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'total_rows': len(df_out),
            'has_actual': has_actual,
            'metrics_if_actual': {
                'mae': mae_prop,
                'rmse': rmse_prop,
                'r2': r2_prop
            } if has_actual else None,
            'rows': results_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
