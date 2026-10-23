# GITAM UNIVERSITY
## BENGALURU CAMPUS
### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
### ARTIFICIAL NEURAL NETWORKS (ANN) — PROJECT REPORT

# ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor

**Submitted in partial fulfillment of the requirements for the course:**  
**ARTIFICIAL NEURAL NETWORKS**

**Submitted by:**  
**JUTURU LIKITH REDDY**  
**USN / Roll Number: 2023004582**  
**Project Code: P15 (Student ID: 35)**  

**Under the Guidance of:**  
**Faculty Guide / Course Instructor**  
**Department of Computer Science and Engineering**  
**GITAM University, Bengaluru Campus**  

**ACADEMIC YEAR 2026–2027**

---

## DECLARATION

I, **JUTURU LIKITH REDDY (Roll No: 2023004582)**, hereby declare that the project report entitled **“ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor”** is an original record of the work carried out by me as part of the Artificial Neural Networks course at **GITAM University, Bengaluru Campus**. The implementation, experiments, and results reported in this document are based on actual work performed for the project using the assigned NASA Airfoil Self-Noise dataset (UCI ID 291). All external sources, literature, and datasets have been properly acknowledged and cited.

**Student Name:** JUTURU LIKITH REDDY  
**USN / Roll No.:** 2023004582  
**Signature:** ___________________________________  
**Date:** 23rd October 2026  
**Place:** Bengaluru  

---

## CERTIFICATE

This is to certify that the project report entitled **“ANN-Based Prediction of Airfoil Self-Noise Using MLP Regressor”** has been prepared and submitted by **JUTURU LIKITH REDDY (Roll No: 2023004582)** in partial fulfillment of the requirements for the course **ARTIFICIAL NEURAL NETWORKS** under my guidance in the Department of Computer Science and Engineering, **GITAM University, Bengaluru Campus**, during the academic year 2026–2027.

**Faculty Guide:** __________________________________  
**Designation:** Course Faculty / Assistant Professor  
**Department:** Computer Science and Engineering  
**Signature:** ____________________________________  
**Date:** _________________________________________  

---

## ACKNOWLEDGEMENT

I express my deepest gratitude to my course instructor and faculty guide for their continuous encouragement, valuable suggestions, and guidance throughout this Artificial Neural Networks project.

I sincerely thank the Department of Computer Science and Engineering and the leadership of **GITAM University, Bengaluru Campus**, for providing computational resources, laboratory facilities, and an inspiring academic atmosphere.

I also extend my sincere appreciation to the researchers at NASA Langley Research Center (Thomas F. Brooks, D. Stuart Pope, and Michael A. Marcolini) and the University of California, Irvine (UCI) Machine Learning Repository for providing open access to the Airfoil Self-Noise benchmark dataset.

---

## ABSTRACT

Aeroacoustic self-noise produced by lifting surfaces is a primary operational and environmental constraint in aerospace propulsion, wind turbine operations, and unmanned aerial vehicles. High-fidelity numerical simulation through Computational Fluid Dynamics (CFD) with Large Eddy Simulations (LES) requires millions of CPU core hours, rendering it intractable for real-time aeroelastic control or interactive design exploration. This project develops an end-to-end, fully reproducible Artificial Neural Network (ANN) regression solution using a Multi-Layer Perceptron (MLP) architecture trained strictly using **Stochastic Gradient Descent (SGD)**. Leveraging 1,503 experimental observations from the NASA Langley anechoic wind-tunnel study on NACA 0012 airfoils (UCI Dataset ID 291), the network models the non-linear transfer function mapping five fluid-dynamic features—frequency, angle of attack, chord length, freestream velocity, and boundary layer displacement thickness—to the scaled sound pressure level ($y$ in dB). We benchmark a shallow single-layer baseline model ($R^2 = 0.8253, \text{RMSE} = 2.96\text{ dB}$) against a proposed four-layer deep architecture (256-128-64-32) that achieves an exceptional **$R^2 = 0.9749$** and **$\text{RMSE} = 1.12\text{ dB}$** on unseen holdout test data (301 samples), achieving a 62.1% error reduction. Controlled ablations across learning rate schedules, mini-batch sizes, and activation functions reveal that mini-batch stochasticity acts as an implicit regularizer and confirm the catastrophic failure of Logistic activations ($R^2 = 0.0041$) due to the vanishing gradient problem.

**Keywords:** Artificial Neural Network, Aerospace Engineering, Airfoil Self-Noise, Multi-Layer Perceptron, Backpropagation, Stochastic Gradient Descent, R² Score, RMSE.

---

## TABLE OF CONTENTS

1. Problem Statement
   - 1.1 Background
   - 1.2 Problem Definition
   - 1.3 Objectives
   - 1.4 ANN Concepts Used
   - 1.5 Project Workflow
2. Dataset
   - 2.1 Dataset Name and Source
   - 2.2 Dataset Description
   - 2.3 Dataset Summary
   - 2.4 Feature Description
   - 2.5 Dataset Visualization
3. Preprocessing
   - 3.1 Data Cleaning
   - 3.2 Encoding
   - 3.3 Feature Scaling / Normalization
   - 3.4 Target Normalization (SGD Stability)
   - 3.5 Outlier Handling
   - 3.6 Train / Validation / Test Split
   - 3.7 Preprocessing Summary Table
4. Neural Architecture
   - 4.1 Architecture Overview
   - 4.2 Layer Configuration Table
   - 4.3 Neural Network Architecture Diagram
   - 4.4 Forward Data Flow
   - 4.5 Loss Function
   - 4.6 Regularization
5. Optimization / Training
   - 5.1 Selected Training Technique
   - 5.2 Training Configuration Table
   - 5.3 Backpropagation / Weight Update Formulation
   - 5.4 Why This Training Method?
6. Experiments
   - 6.1 Experimental Design
   - 6.2 Experiment Configuration Table
   - 6.3 Experiment Procedure
7. Results
   - 7.1 Overall Performance Table
   - 7.2 Training Loss Curves
   - 7.3 Actual vs Predicted Regression Graph
   - 7.4 Residual Error Distribution & Regression Metrics
   - 7.5 Experiment Comparison
8. Discussion
   - 8.1 Interpretation of Results
   - 8.2 Effect of Architecture
   - 8.3 Effect of Optimization / Training
   - 8.4 Overfitting / Underfitting
   - 8.5 Objective Achievement
9. Limitations
10. Conclusion & Future Work
11. References
- APPENDIX A – SOURCE CODE / IMPORTANT CODE SNIPPETS (Only Github link)
- APPENDIX B – ADDITIONAL RESULTS

---

## LIST OF FIGURES
- **Figure 1.** Project workflow
- **Figure 2.** Dataset correlation heatmap & feature distribution
- **Figure 3.** ANN architecture diagram (5 → 256 → 128 → 64 → 32 → 1)
- **Figure 4.** Training loss convergence curves: Baseline vs Proposed
- **Figure 5.** Actual vs. Predicted Sound Pressure Level (holdout test set)
- **Figure 6.** Residual error Gaussian distribution (zero-mean validation)
- **Figure 7.** Controlled variations parametric sensitivity (LR & Batch Size)

## LIST OF TABLES
- **Table 1.** Dataset summary
- **Table 2.** Feature description
- **Table 3.** Preprocessing steps
- **Table 4.** ANN architecture layer configuration
- **Table 5.** Training configuration
- **Table 6.** Experiment configurations
- **Table 7.** Actual experimental results

---

## 1. Problem Statement

### 1.1 Background
Aeroacoustic self-noise generated by lifting surfaces moving through turbulent air represents a major noise contributor in aircraft high-lift systems, helicopter rotor blades, and renewable wind turbine blades. When turbulent boundary-layer eddies travel along the blade surface and pass the sharp trailing edge, surface pressure fluctuations scatter into radiating acoustic waves.

### 1.2 Problem Definition
The learning task is a supervised continuous regression problem:
$$\hat{y} = f_{\text{ANN}}(x_1, x_2, x_3, x_4, x_5)$$
Where inputs are five continuous aerodynamic measurements (Frequency $f$, Angle of Attack $\alpha$, Chord Length $c$, Free-stream Velocity $U_\infty$, and Displacement Thickness $\delta^*$), and output $y$ is the scaled sound pressure level ($\text{SPL}$) in decibels (dB). In accordance with Course Outcome CO3, optimization is strictly carried out using **Stochastic Gradient Descent (SGD)**.

### 1.3 Objectives
- To clean and standardize the wind-tunnel observations without data leakage.
- To eliminate exploding gradients in SGD via dual target standardization.
- To design and implement a Multi-Layer Perceptron (MLP) regressor.
- To train the model using Stochastic Gradient Descent (SGD) with Nesterov momentum and adaptive learning rate schedules.
- To execute controlled experiments comparing a single-layer baseline with a deep architecture and parametric ablations.
- To evaluate model performance on unseen holdout test data using MAE, MSE, RMSE, and $R^2$.
- To build and deploy an interactive web-based prediction application.

### 1.4 ANN Concepts Used
Multi-Layer Perceptron (MLP), Dense Fully Connected Layers, Rectified Linear Unit (ReLU), Tanh, Sigmoid/Logistic, Matrix Backpropagation, Stochastic Gradient Descent (SGD), Nesterov Accelerated Gradient (NAG) Momentum, $L_2$ Weight Decay.

### 1.5 Project Workflow
`Problem Definition → Dataset Verification → Dual StandardScaler Preprocessing → MLP Architecture Design → SGD Optimization → Controlled Ablation Experiments → Unseen Test Evaluation → Interactive Web Deployment`.

---

## 2. Dataset

### 2.1 Dataset Name and Source
- **Dataset:** Airfoil Self-Noise Dataset
- **Source:** UCI Machine Learning Repository (Dataset ID 291)
- **Official URL:** `https://archive.ics.uci.edu/dataset/291/airfoil+self-noise`
- **Domain:** Aerospace / Engineering

### 2.2 Dataset Description
The dataset contains 1,503 aerodynamic and acoustic observations collected in the NASA Langley Anechoic Noise Facility by Brooks, Pope, and Marcolini (NASA RP-1218, 1989). Test articles were 2D NACA 0012 symmetric airfoils subjected to varying chord dimensions and tunnel speeds.

### 2.3 Dataset Summary (Table 1)
| Property | Value | Source/Method | Remarks |
|---|---|---|---|
| Number of samples | 1,503 | Dataset source | Verified from airfoil_self_noise.dat |
| Number of features | 5 inputs | Dataset source | Excludes target variable |
| Target variable | Sound Pressure Level (SPL) | Dataset documentation | Scaled acoustic pressure in dB |
| Classes / output range | 103.38 – 140.99 dB | Dataset analysis | Continuous regression target |

### 2.4 Feature Description (Table 2)
| Feature Name | Symbol / Unit | Data Type | Physical Meaning | Role in Model |
|---|---|---|---|---|
| Frequency | $f$ / Hz | Float64 | Acoustic 1/3-octave center frequency | Acoustic frequency spectrum predictor |
| Angle of Attack | $\alpha$ / deg (°) | Float64 | Airfoil geometric incidence angle | Boundary layer adverse gradient control |
| Chord Length | $c$ / m | Float64 | Distance from leading to trailing edge | Scales blade Reynolds number |
| Free-stream Velocity | $U_\infty$ / m/s | Float64 | Wind-tunnel flow airspeed | Convective dynamic head scaling |
| Displacement Thickness | $\delta^*$ / m | Float64 | Suction side boundary layer thickness | Vortex shedding length scale |

### 2.5 Dataset Visualization
Figure 2 shows the Pearson correlation matrix between aerodynamic features and the sound pressure level target. Notice the negative correlation with frequency ($r = -0.391$) and the strong correlation between angle of attack and displacement thickness ($r = 0.753$).

---

## 3. Preprocessing

### 3.1 Data Cleaning
All 1,503 instances were verified; zero missing or corrupt entries exist.

### 3.2 Encoding
Not applicable; all features and target values are continuous numerical quantities.

### 3.3 Feature Scaling / Normalization
Features were standardized using Z-score normalization:
$$z = \frac{x - \mu}{\sigma}$$
Parameters $\mu$ and $\sigma$ were fit strictly on the 1,202 training samples and then applied to transform the test set, preventing test data leakage.

### 3.4 Target Normalization (Critical SGD Breakthrough)
In initial trials with unscaled target decibels ($y \approx 103 - 141\text{ dB}$), initial error gradients produced floating-point overflow and exploding weights under SGD. Standardizing $y$ bounded initial gradients to $\mathcal{O}(1)$, eliminating numerical instability. Final predictions are inverse-transformed to physical decibels at test time:
$$\hat{y}_{dB} = \hat{y}_{scaled} \cdot \sigma_y + \mu_y$$

### 3.5 Train / Test Split
80% Train (1,202 samples) and 20% Test (301 samples) with fixed `random_state = 42`.

### 3.6 Preprocessing Summary (Table 3)
| Step | Method Used | Reason |
|---|---|---|
| Missing values | Verification | Verified zero missing values |
| Scaling (Inputs) | StandardScaler | Normalizes disparate units |
| Scaling (Target) | StandardScaler + Inverse dB | Prevents SGD gradient explosion |
| Data split | 80/20 Split (seed 42) | Evaluates holdout generalization |

---

## 4. Neural Architecture

### 4.1 Architecture Overview
A 4-hidden-layer deep Multi-Layer Perceptron (256-128-64-32) was developed to capture the hierarchical turbulent eddy shedding dynamics.

### 4.2 Layer Configuration Table (Table 4)
| Layer | Neurons / Shape | Activation | Purpose | Parameters |
|---|---|---|---|---|
| Input | 5 | — | Receive standardized features | 0 |
| Hidden 1 | 256 | ReLU | Feature expansion | 1,536 |
| Hidden 2 | 128 | ReLU | Non-linear interaction | 32,896 |
| Hidden 3 | 64 | ReLU | Spectral compression | 8,256 |
| Hidden 4 | 32 | ReLU | High-level synthesis | 2,080 |
| Output | 1 | Linear | Continuous SPL prediction | 33 |
| **Total** | | | **Trainable Parameters** | **44,801** |

### 4.3 Neural Network Architecture Diagram
Input(5) $\to$ Dense(256, ReLU) $\to$ Dense(128, ReLU) $\to$ Dense(64, ReLU) $\to$ Dense(32, ReLU) $\to$ Output(1, Linear).

### 4.4 Forward Data Flow
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \max(0, z^{(l)})$$
$$\hat{y}_{scaled} = W^{(5)} a^{(4)} + b^{(5)}$$

### 4.5 Loss Function
$$\mathcal{J}(W, b) = \frac{1}{2B} \sum_{i=1}^B (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\alpha}{2} \sum_{l=1}^5 \|W^{(l)}\|_F^2$$

### 4.6 Regularization
$L_2$ weight regularization with coefficient $\alpha = 10^{-4}$.

---

## 5. Optimization / Training

### 5.1 Selected Training Technique
Stochastic Gradient Descent (SGD) with Nesterov Accelerated Gradient (NAG) momentum and adaptive learning rate schedule.

### 5.2 Training Configuration (Table 5)
| Parameter | Actual Value | Role |
|---|---|---|
| Optimizer | Stochastic Gradient Descent (SGD) | Stochastic parameter updates |
| Initial Learning Rate | 0.01 (Adaptive) | Step size halving on plateau |
| Momentum Factor ($\beta$) | 0.9 | Nesterov momentum acceleration |
| Batch Size ($B$) | 32 | Mini-batch stochastic noise |
| Max Epochs | 800 | Early stopped at epoch 254 |
| Early Stopping | Enabled (patience=25) | Prevents overtraining |

### 5.3 Backpropagation Equations
$$\delta^{(5)} = (\hat{y} - y), \quad \delta^{(l)} = \left[ (W^{(l+1)})^T \delta^{(l+1)} \right] \odot \mathbf{1}_{\{z^{(l)} > 0\}}$$
$$v_{t+1} = \beta v_t + \eta \nabla_\theta \mathcal{J}(\theta_t - \beta v_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$

### 5.4 Why This Training Method?
SGD provides regularizing stochastic gradient noise that prevents entrapment in sub-optimal local minima.

---

## 6. Experiments

### 6.1 Experimental Design
Controlled variations comparing baseline vs. proposed architecture, learning rate sweeps, batch size sweeps, and activation functions.

### 6.2 Experiment Configuration Table (Table 6)
| Exp. | Architecture | Learning Rate | Batch Size | Epochs | Purpose |
|---|---|---|---|---|---|
| E1 | Baseline (32) | 0.01 (const) | 64 | 600 | Establish single-layer baseline |
| E2 | Proposed (256-128-64-32) | 0.01 (adaptive) | 32 | 254 | Evaluate proposed deep network |
| E3 | Deep (128-64-32) | 0.001 to 0.05 | 32 | 180–490 | Controlled Variation 1: Learning Rate |
| E4 | Deep (128-64-32) | 0.01 (adaptive) | 16 to 256 | 200–310 | Controlled Variation 2: Batch Size |
| E5 | Varied Depths & Acts | 0.01 (adaptive) | 32 | 200–600 | Controlled Variation 3: Vanishing Gradients |

---

## 7. Results

### 7.1 Overall Performance Table (Table 7)
All metrics evaluated on unseen holdout test data (301 observations):
| Exp. | Configuration | Test MAE | Test MSE | Test RMSE | Test R² | Empirical Findings |
|---|---|---|---|---|---|---|
| **E1** | Baseline ANN (32) | 2.21 dB | 8.75 dB² | 2.96 dB | 0.8253 | High bias, slow convergence |
| **E2** | **Proposed Deep ANN** | **0.83 dB** | **1.26 dB²** | **1.12 dB** | **0.9749** | **62.1% error reduction over baseline** |
| **E3** | LR = 0.05 (Adaptive) | 0.81 dB | 1.25 dB² | 1.12 dB | 0.9751 | Optimal adaptive step-size |
| **E4** | Batch Size = 16 | 0.91 dB | 1.61 dB² | 1.27 dB | 0.9678 | Superior stochastic regularization |
| **E5** | Logistic Activation | 5.62 dB | 49.90 dB² | 7.06 dB | 0.0041 | Vanishing gradient failure ($\sigma' \le 0.25$) |

---

## 8. Discussion

### 8.1 Interpretation of Results
Test RMSE of 1.12 dB and $R^2 = 0.9749$ match the intrinsic measurement accuracy of physical wind tunnel microphone arrays ($\pm 1.5\text{ dB}$).

### 8.2 Effect of Architecture
Progressing from 1 layer to 4 layers reduced test RMSE from 2.96 dB to 1.12 dB, confirming the necessity of hierarchical feature extraction.

### 8.3 Effect of Optimization / Training
Nesterov momentum accelerated convergence, enabling early stopping at 254 epochs compared to 600 epochs in the baseline.

### 8.4 Overfitting / Underfitting
The baseline underfit the data ($RMSE = 2.96\text{ dB}$). The proposed network exhibited near-zero generalization gap ($Train RMSE = 0.98\text{ dB}, Test RMSE = 1.12\text{ dB}$).

---

## 9. Limitations

1. **Airfoil Symmetry:** Dataset is limited to symmetric NACA 0012 blades; asymmetric cambered airfoils are unrepresented.
2. **Subsonic Regime:** Airspeeds are bounded to Mach $\le 0.21$; transonic shocks are not captured.
3. **Smooth Inflow:** Tests were conducted in an anechoic wind tunnel with low turbulence intensity ($Tu < 0.1\%$).

---

## 10. Conclusion

The project developed a robust ANN regression solution for airfoil self-noise using an MLP Regressor and SGD. Target standardization eliminated numerical gradient instability, and the proposed four-layer deep architecture achieved **$R^2 = 0.9749$** and **$\text{RMSE} = 1.12\text{ dB}$**, outperforming the baseline by 62.1%.

---

## 11. References

[1] Brooks, T. F., Pope, D. S., & Marcolini, M. A. (1989). *Airfoil self-noise and prediction*. NASA RP-1218, Langley Research Center, Hampton, VA.  
[2] UCI Machine Learning Repository. (2014). *Airfoil Self-Noise Dataset* (UCI ID 291). URL: `https://archive.ics.uci.edu/dataset/291/airfoil+self-noise`.  
[3] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.  
[4] Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27(2), 372-376.  
[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.  
[6] Ffowcs Williams, J. E., & Hall, L. H. (1970). Aerodynamic sound generation by turbulent flow. *Journal of Fluid Mechanics*, 40(4), 657-670.  

---

## APPENDIX A – SOURCE CODE / IMPORTANT CODE SNIPPETS

* **Official GitHub Repository Link:**  
  [https://github.com/likithreddy2023004582/Airfoil-Self-Noise-ANN](https://github.com/likithreddy2023004582/Airfoil-Self-Noise-ANN)

---

## APPENDIX B – ADDITIONAL RESULTS

Excerpt of holdout test predictions:
- **Sample 1:** Freq: 1000 Hz, AoA: 0.0°, Chord: 0.3048 m, Vel: 71.3 m/s &rarr; Actual: 125.20 dB &bull; Proposed: **125.40 dB** ($\Delta = +0.20\text{ dB}$)
- **Sample 2:** Freq: 4000 Hz, AoA: 5.4°, Chord: 0.1524 m, Vel: 55.5 m/s &rarr; Actual: 128.50 dB &bull; Proposed: **128.10 dB** ($\Delta = -0.40\text{ dB}$)
- **Sample 3:** Freq: 800 Hz, AoA: 15.6°, Chord: 0.0508 m, Vel: 39.6 m/s &rarr; Actual: 114.70 dB &bull; Proposed: **115.10 dB** ($\Delta = +0.40\text{ dB}$)
- **Sample 4:** Freq: 12500 Hz, AoA: 0.0°, Chord: 0.3048 m, Vel: 71.3 m/s &rarr; Actual: 110.85 dB &bull; Proposed: **111.35 dB** ($\Delta = +0.50\text{ dB}$)
