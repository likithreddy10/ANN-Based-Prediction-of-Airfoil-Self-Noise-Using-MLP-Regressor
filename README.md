# ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor

**Student Name:** JUTURU LIKITH REDDY  
**Registration Number:** 2023004582  
**Project Code:** P15 / 35  
**Course Outcome:** Unit 3 / CO3  
**Domain:** Aerospace / Engineering  
**Dataset:** Airfoil Self-Noise (UCI ID 291)  
**Algorithm:** Multi-Layer Perceptron (MLP) trained with Stochastic Gradient Descent (SGD)  

---

## 🚀 Quick Start Guide

### 1. Launch the Interactive Web Application
You can launch the web application in any of the following ways:
- **Option A:** Double click `start_website.bat`
- **Option B:** Run in terminal / PowerShell:
  ```bash
  python app.py
  ```
  Then open your browser at **[http://localhost:5000](http://localhost:5000)**.

### 2. Run the Verification Script
To run an instant CLI test that verifies the models, makes sample predictions, and evaluates on the 301 test samples:
```bash
python run_demo.py
```

### 3. Re-train All Controlled Experiments from Scratch
To re-run the full training pipeline (baseline, proposed deep architecture, learning rate sweep, batch size sweep, and architecture depth sweep) and regenerate all plots:
```bash
python train_experiments.py
```

---

## 🌟 Web Application Features

1. **Prediction Studio (`/`):**
   - Synchronized sliders and numeric inputs for all 5 aerodynamic parameters.
   - Live real-time acoustic inference with side-by-side comparison between Baseline and Proposed models.
   - Dynamic Sound Pressure Level (dB) visual gauge and noise intensity categorization.
   - Physical non-dimensional indicators: Strouhal Number ($St_{\delta^*}$), Chord Reynolds Number ($Re_c$), Mach Number ($M$), and Acoustic Wavelength ($\lambda$).
   - Dynamic 1/3-octave frequency spectrum curve (200 Hz – 20,000 Hz) plotted using Chart.js.

2. **Controlled Experiments & Ablations (`/experiments`):**
   - Direct empirical comparison of Baseline ($R^2 = 0.8253$) vs Proposed ($R^2 = 0.9749$).
   - Controlled Experiment 1: Learning Rate Variations ($\eta \in [0.001, 0.05]$).
   - Controlled Experiment 2: Mini-Batch Size Variations ($B \in [16, 256]$).
   - Controlled Experiment 3: Depth and Activation Functions (ReLU vs Tanh vs Logistic showing vanishing gradients).
   - High-resolution loss convergence, residual distribution, and actual vs predicted scatter plots.

3. **Aerodynamic Dataset Explorer (`/dataset`):**
   - Summary statistics (min, mean, std, max) for all variables.
   - Aerodynamic Pearson correlation heatmap.
   - Aeroacoustic physical mechanism breakdowns (frequency decay, $\alpha$ vs $\delta^*$ coupling).
   - Interactive observation table preview with sample CSV download.

4. **Neural Architecture & Mathematics (`/architecture`):**
   - Layer-by-layer tensor shapes and parameter count breakdown (44,801 trainable parameters).
   - Analytical formulation of forward pass, Mean Squared Error loss with $L_2$ regularization, backpropagation chain rule, and Nesterov Accelerated Gradient (NAG) momentum SGD updates.
   - Mathematical derivation of the target scaling breakthrough that eliminated exploding gradients.

5. **Batch CSV Prediction (`/batch`):**
   - Upload custom CSV observation files.
   - Simultaneous batch inference through both models.
   - Automatic calculation of test metrics (MAE, RMSE, $R^2$) when ground truth is provided.
   - Export predictions as augmented CSV.

6. **10–15 Page Academic Report (`/report` & `PROJECT_REPORT.md`):**
   - Complete formal academic paper following the exact required flow:
     *Problem Statement &rarr; Dataset &rarr; Preprocessing &rarr; Neural Architecture &rarr; Optimization/Training &rarr; Experiments &rarr; Results &rarr; Discussion &rarr; Limitations &rarr; Conclusion &rarr; References*.
   - Specialized CSS `@media print` styling for 1-click PDF generation via `window.print()`.

---

## 📁 Repository Structure

```
ANN_LAB/
│
├── airfoil_self_noise.dat          # UCI 291 aerodynamic raw dataset (1,503 rows)
├── app.py                          # Flask web application & REST APIs
├── train_experiments.py            # Complete controlled experiments training suite
├── run_demo.py                     # Standalone CLI validation demonstration script
├── requirements.txt                # Python package dependencies
├── start_website.bat               # Windows 1-click batch launcher
├── PROJECT_REPORT.md               # 10–15 page formal academic markdown report
├── README.md                       # Repository guide and overview
│
├── model_artifacts/
│   ├── baseline_model.joblib       # Trained Baseline ANN model (32 dense)
│   ├── proposed_model.joblib       # Trained Proposed Deep ANN (256-128-64-32)
│   ├── scaler_x.joblib             # Feature standardizer
│   ├── scaler_y.joblib             # Target standardizer
│   └── experiments_results.json    # Exact empirical metrics and loss curves
│
├── static/
│   ├── charts/
│   │   ├── actual_vs_predicted.png      # Scatter plot of holdout predictions
│   │   ├── residual_distribution.png    # Residual error Gaussian distribution
│   │   ├── loss_curves_comparison.png   # SGD loss convergence curves
│   │   ├── correlation_heatmap.png      # Pearson correlation matrix
│   │   └── controlled_variations.png    # LR & Batch size sensitivity curves
│   ├── css/
│   └── js/
│
└── templates/
    ├── base.html                   # Master layout with aerospace theme & navbar
    ├── index.html                  # Homepage & Prediction Studio with spectrum plot
    ├── experiments.html            # Controlled experiments & ablation tables
    ├── dataset.html                # Dataset explorer & feature statistics
    ├── architecture.html           # Neural network topology & SGD mathematics
    ├── batch.html                  # Batch CSV inference engine
    └── report.html                 # Complete printable academic report
```

---

## 📊 Summary of Actual Experimental Results

| Model / Configuration | Architecture | Optimizer | Epochs | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| **Baseline ANN** | Dense(32) | SGD (const 0.01) | 600 | 2.21 dB | 2.96 dB | 0.8253 |
| **Proposed Deep ANN** | Dense(256-128-64-32) | SGD + NAG (0.9) | 254 | **0.83 dB** | **1.12 dB** | **0.9749** |
| *LR = 0.001* | Dense(128-64-32) | SGD (adaptive) | 492 | 1.28 dB | 1.70 dB | 0.9425 |
| *LR = 0.050* | Dense(128-64-32) | SGD (adaptive) | 189 | 0.81 dB | 1.12 dB | 0.9751 |
| *Batch = 16* | Dense(128-64-32) | SGD (adaptive) | 212 | 0.91 dB | 1.27 dB | 0.9678 |
| *Batch = 256* | Dense(128-64-32) | SGD (adaptive) | 305 | 1.21 dB | 1.70 dB | 0.9424 |
| *Logistic Activation* | Dense(128-64-32) | SGD (adaptive) | 600 | 5.62 dB | 7.06 dB | 0.0041 *(Vanishing Gradient)* |

**Key Academic Takeaways:**
1. Dual input and target standardization solved the exploding gradient problem under SGD.
2. The proposed deep architecture reduced test RMSE by **62.1%** compared to the baseline.
3. Smaller mini-batches ($B = 16, 32$) demonstrated superior test generalization due to stochastic regularizing noise.
4. Logistic activations failed to learn ($R^2 \approx 0.004$) due to derivative saturation ($\sigma'(z) \le 0.25$), empirically proving the vanishing gradient problem.
