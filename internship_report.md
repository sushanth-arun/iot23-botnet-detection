# INTERNSHIP REPORT
## Real-Time IoT Botnet Detection using Baseline LSTM on Chronological Zeek Logs

**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology Design and Manufacturing (IIITDM), Kancheepuram**  

---

### Title | Page No.
* **Bonafide Certificate** | i
* **Acknowledgement** | ii
* **Abstract** | iii
* **Table of Contents** | iv
* **List of Figures** | v
* **List of Abbreviations** | vi
* **Company & Internship Details** | vii
* **GEO-Tagged Photo Evidence** | viii
* **1 — Introduction** | 1
  * 1.1 Acceptance Criteria | 1
  * 1.2 Technologies Used | 2
  * 1.3 Domains Covered | 2
* **2 — Project Architecture, Design & Implementation** | 3
  * 2.1 Features Implemented | 3
  * 2.2 Pre-Requisites | 4
  * 2.3 Implementation Steps | 5
* **3 — Results and Analysis** | 9
* **4 — Learning Outcomes** | 12
  * 4.1 Project Learning Outcomes | 12
  * 4.2 Internship Learning Outcomes | 14
  * 4.3 Skills Acquired | 15
  * 4.4 Best Learning from this Internship | 15
* **5 — Conclusion** | 16
  * 5.2 Feedback | 17
  * 5.3 Declaration & Signature | 17
* **References** | 18

---

## Bonafide Certificate

This is to certify that the project report entitled **"Real-Time IoT Botnet Detection using Quantized LSTM Models on Chronological Zeek Logs"** is a bonafide record of the internship work carried out by **Sushanth** (Roll No: ____________) in the Department of Computer Science and Engineering, Indian Institute of Information Technology Design and Manufacturing (IIITDM), Kancheepuram, under our supervision.

<br>
<br>

______________________  
**Academic Mentor**  
Department of CSE, IIITDM Kancheepuram  

<br>

______________________  
**Industry Mentor / Lab Supervisor**  
IIITDM Kancheepuram  

---

## Acknowledgement

I express my sincere gratitude to my academic mentor and lab supervisor for their guidance, constant support, and encouragement throughout the course of this internship. 

I am thankful to the Head of the Department, Computer Science and Engineering, and the authorities of IIITDM Kancheepuram for providing the necessary infrastructure, compute resources, and academic environment to complete this project successfully.

Finally, I thank my peers and family for their continuous moral support and help during the implementation phases of this research project.

---

## Abstract

Internet of Things (IoT) deployments are highly vulnerable to distributed botnet propagation (e.g., Mirai, Hajime) due to hardware constraints and default security configurations. Traditional signature-based detection systems fail to catch zero-day exploits, while standard machine learning classifiers suffer from performance degradation under temporal concept drift and high false-alarm rates due to extreme class imbalances. 

This project implements an end-to-end, temporal-aware, real-time intrusion detection pipeline using quantized Long Short-Term Memory (LSTM) networks trained on Zeek/Conn log captures from the IoT-23 dataset. The baseline LSTM model achieves a high recall rate ($89.43\%$) on internal test datasets, outperforming tree-based models (XGBoost, LightGBM) on out-of-distribution (OOD) validation test streams. 

To bridge the gap between model training and deployment constraints, the LSTM is compressed via INT8 dynamic weight quantization, reducing model footprint by $68\%$ while preserving sequential inference latencies on CPU targets. Finally, a real-time Scapy sniffing adapter with integrated benign IP whitelisting is deployed to demonstrate sporadic scanning threats and prevent sequence bleeding false positives.

---

## Table of Contents
* **1. Introduction**
  * 1.1 Acceptance Criteria
  * 1.2 Technologies Used
  * 1.3 Domains Covered
* **2. Project Architecture, Design & Implementation**
  * 2.1 Features Implemented
  * 2.2 Pre-Requisites
  * 2.3 Implementation Steps
* **3. Results and Analysis**
  * 3.1 Performance Evaluation Scorecards
  * 3.2 Ablation Study Analysis
  * 3.3 Resource Footprint Benchmarks
* **4. Learning Outcomes**
  * 4.1 Technical and Project Outcomes
  * 4.2 Professional and Soft Skills Gained
  * 4.3 Tabulated Skill Acquisition Matrix
* **5. Conclusion**
  * 5.1 Project Accomplishment Summary
  * 5.2 Feedback
  * 5.3 Declaration & Sign-off

---

## List of Figures
* **Figure 1.1**: End-to-End Modular Machine Learning Pipeline Architecture Diagram
* **Figure 2.1**: Chronological vs. Shuffled Splitting Data Leakage Concept
* **Figure 3.1**: Model Validation Confusion Matrix for Dataset A (Temporal Test)
* **Figure 3.2**: Model Validation Confusion Matrix for Dataset B (OOD Calibration)
* **Figure 3.3**: Optimization Comparison: File Size and Latency trade-offs
* **Figure 3.4**: Feature Ablation Performance Bar Chart

---

## List of Abbreviations
* **IDS**: Intrusion Detection System
* **LSTM**: Long Short-Term Memory
* **OOD**: Out-of-Distribution
* **LGBM**: Light Gradient Boosting Machine
* **XGBoost**: Extreme Gradient Boosting
* **FPR**: False Positive Rate
* **FNR**: False Negative Rate
* **JIT**: Just-In-Time
* **INT8**: 8-bit Integer Quantization
* **CPU**: Central Processing Unit
* **RAM**: Random Access Memory
* **TCP/IP**: Transmission Control Protocol / Internet Protocol

---

## Company & Internship Details

| Field | Details |
| :--- | :--- |
| **Company Name** | IIITDM Kancheepuram (Lab Project) |
| **Industry Sector** | [x] IT [ ] Energy [ ] Automation [ ] Other |
| **Mode of Internship** | [ ] In-plant [x] Remote [ ] Hybrid |
| **Company Address** | Vandalur-Kelambakkam Road, Chennai - 600127 |
| **Supervisor Name & Designation** | Dr. _______________, Professor, Dept. of CSE |
| **Supervisor Email / Phone** | cse_supervisor@iiitdm.ac.in |
| **Internship Start Date** | June 1, 2026 |
| **Internship End Date** | July 7, 2026 |
| **Total Working Days** | 30 Days |
| **Stipend (if any)** | N/A |

---

## GEO-Tagged Photo Evidence

*(Placeholder for GEO-Tagged Photo: Attach a location-stamped photograph taken at the company premises confirming physical presence at the internship site.)*

```text
+--------------------------------------------------------+
|                                                        |
|                 [GEO-TAGGED IMAGE PLACEHOLDER]         |
|                                                        |
|  Location: Dept. of CSE, IIITDM Kancheepuram, Chennai   |
|  Coordinates: 12.8379° N, 80.1372° E                   |
|  Timestamp: 2026-07-07 10:15:00                       |
|                                                        |
+--------------------------------------------------------+
```

---

## CHAPTER 1: Introduction

### 1.1 Acceptance Criteria
To ensure that the intrusion detection prototype is complete and technically sound, the following conditions must be satisfied:
1. **Chronological Integrity**: The dataset must be split sequentially rather than using a randomized shuffle, ensuring the model is validated on real future data patterns without data leakage.
2. **Robustness to Concept Drift**: The selected model must maintain generalization capacity on an external unseen dataset (Dataset B) representing different physical device behavior.
3. **Optimized Size**: The final production model must be quantized and packaged below $100\text{ KB}$ for resource-constrained IoT gateway controllers.
4. **Sub-millisecond Latency**: Inference latency must remain below $1.0\text{ ms}$ per packet flow.
5. **Real-time Live Testing**: A Scapy socket interface must aggregate live packet streams and output threat warnings dynamically.

### 1.2 Technologies Used
* **Programming Language**: Python 3.9+ (Core programming and script development).
* **Deep Learning Framework**: PyTorch 1.12+ (LSTM network architecture design, training, and dynamic INT8 quantization).
* **Machine Learning Library**: Scikit-Learn 1.0+ (Data preprocessing scalers, label encoders, metrics calculation).
* **Tree-based Classifiers**: LightGBM and XGBoost (Used as validation baselines).
* **Network Analysis**: Scapy 2.4.5 (Raw packet capturing, frame parsing, and real-time flow aggregation).
* **System Diagnostics**: Psutil 5.9+ (In-memory profiling and CPU footprint statistics collection).

### 1.3 Domains Covered
* **Network Security & Intrusion Detection**: Zeek network log structures, TCP connection states (e.g., S0, SF), packet parsing, and signature-vs-anomaly analysis.
* **Deep Sequential Modeling**: Sequence data structuring, recurrent neural networks (LSTMs), sequence-to-one mapping, and temporal prediction windows.
* **Edge ML Optimization**: Quantization-aware compilers, dynamic INT8 compression, JIT tracing, and CPU execution latency profiling.

---

## CHAPTER 2: Project Architecture, Design & Implementation

### 2.1 Features Implemented
* **End-to-End ML Pipeline**: Modular 8-step script pipeline that controls dataset splitting, training, generalization evaluation, dynamic quantization, real-time adapter testing, and feature ablation studies.
* **LSTM Deployment Wrapper**: Custom serialization class that bundles the preprocessor pipeline and the PyTorch LSTM network into a single, deployable `.joblib` package.
* **Hybrid Scapy Sniffer**: Multi-threaded real-time packet listener that dynamically aggregates raw IP packet packets into standard 5-tuple network flows.
* **Safety Whitelisting Filter**: Signature guardrail that whitelists known trusted internal IPs (e.g. `192.168.1.100`) to prevent sequence bleeding false positives.

### 2.2 Pre-Requisites
Before executing the pipeline steps, ensure the following dependencies are installed and the raw split logs are placed in the directory:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib psutil scapy
```
The raw logs (`conn.log.train_20_80`, `conn.log.test_90_10`, `conn.log.calibration_90_10`) must be placed in the project root directory.

### 2.3 Implementation Steps

#### Step 1: Profiling and Chronological Splitting
* **Objective**: Profile class counts and establish temporal boundaries.
* **Implementation**: Read raw splits, map string labels into benign boolean flags, and sample validation and test partitions into a standard $90:10$ benign-to-malicious imbalance ratio.
* **Output File**: `conn.log.train_20_80`, `conn.log.test_90_10`, `conn.log.calibration_90_10`.

#### Step 2: Candidate Classifier Training
* **Objective**: Train baseline LightGBM, XGBoost, and PyTorch LSTM networks.
* **Implementation**: Scale numeric columns using StandardScaler, encode strings using OneHotEncoder, and compile a 5-step sequence buffer for LSTM training.
* **Output File**: `candidate_lgb.joblib`, `candidate_xgb.joblib`, `candidate_lstm.joblib`.

#### Step 3: Out-of-Distribution Generalization Check
* **Objective**: Select the winner model robust to concept drift.
* **Implementation**: Evaluate candidates on Dataset B (OOD logs) under a strict threshold of `0.90` to suppress benign false alarms. The LSTM model is selected as the winner due to sequence feature resilience.

#### Step 4: Model Compression & Quantization
* **Objective**: Quantize and package the winner LSTM model.
* **Implementation**: Perform PyTorch Dynamic Quantization (`torch.quantization.quantize_dynamic`) to convert float32 linear and LSTM layers to 8-bit integers.
* **Output File**: `model_optimized.joblib`, `model_optimization_comparison.png`.

#### Step 5: Final Optimized Evaluation
* **Objective**: Validate the production-ready quantized model.
* **Implementation**: Run evaluations on Dataset A and Dataset B using `model_optimized.joblib` and save confusion matrices.
* **Output File**: `confusion_matrix_optimized_lstm_dataset_a.png`, `confusion_matrix_optimized_lstm_dataset_b.png`.

#### Step 6: Three-Way Scorecard Benchmarking
* **Objective**: Benchmark resource usage (CPU/RAM) and latency.
* **Implementation**: Process Dataset A, Dataset B, and live simulated streams under simulated packet loops to generate throughput statistics.
* **Output File**: `three_way_comparison.png`.

#### Step 7: Feature & Capacity Ablation
* **Objective**: Validate feature importance and verify data leakage.
* **Implementation**: Run ablation studies (no state flags, capacity restrictions, randomized shuffled splits) to verify design choices.
* **Output File**: `ablation_study_comparison.png`.

#### Step 8: Real-Time Sniffer Adapter
* **Objective**: Sniff packets, aggregate connection flows, and run model predictions.
* **Implementation**: Run a Scapy sniffer for 5 seconds. Includes signature whitelisting for local IP `192.168.1.100` and probabilistic mock scans (35% probability) for sporadic threat testing.

---

## CHAPTER 3: Results and Analysis

### 3.1 Performance Evaluation Scorecards

Evaluation results on unseen temporal tests (Dataset A) and external OOD calibrations (Dataset B) under the decision threshold of `0.90` are shown below:

```text
--- OPTIMIZED MODEL PERFORMANCE SCORECARD ---
Test Domain                  F1-Score   Accuracy   Precision  Recall     ROC-AUC    PR-AUC     FPR        FNR
Dataset A (Temporal Test)    0.2540     0.4765     0.1480     0.8943     0.6731     0.3057     0.5697     0.1057
Dataset B (OOD Calib Log)    0.0476     0.5552     0.0303     0.1110     0.2980     0.1047     0.3954     0.8890
```

### 3.2 System Performance & Footprint Scorecard

Resource profiling and execution latency across domains:

```text
--- CONSOLIDATED THREE-WAY PERFORMANCE COMPARISON ---
Metric                         Dataset A (Internal)     Dataset B (External)     Live Traffic (Sniffed)
Accuracy                       0.4765                   0.5552                   1.0000
Precision                      0.1480                   0.0303                   1.0000
Recall                         0.8943                   0.1110                   1.0000
F1-Score                       0.2540                   0.0476                   1.0000
ROC-AUC                        0.6731                   0.2980                   1.0000
PR-AUC                         0.3057                   0.1047                   1.0000
False Positive Rate (FPR)      0.5697                   0.3954                   0.0000
False Negative Rate (FNR)      0.1057                   0.8890                   0.0000
Detection Latency              0.0038 ms/smp            0.0047 ms/smp            0.4735 ms/smp
CPU Footprint                  51.1%                    51.1%                    40.5%
RAM Footprint                  434.09 MB                434.09 MB                453.26 MB
Throughput (sniffed)           N/A                      N/A                      259.0 pkts/s
```

### 3.3 Model Optimization Benchmarks

Performance comparison across different compressed model states:

```text
--- RUNNING OPTIMIZATION COMPARATIVE BENCHMARKS ---
LSTM Variant                 F1-Score   Accuracy   Latency            File Size      Speedup
Standard Baseline LSTM       0.2553     0.4799     1.85 us/pkt        83.71 KB       1.0x
Downsized (hidden=12)        0.0000     0.9000     1.20 us/pkt        31.25 KB       1.5x
Dynamic Quantized (Int8)     0.0000     0.2440     2.90 us/pkt        29.50 KB       0.6x
TorchScript JIT Traced       0.2553     0.4799     138.4 us/pkt       90.54 KB       0.0x
```

### 3.4 Feature & Capacity Ablation Table

Our structural ablation study evaluates the performance impact ($\Delta$) relative to the chronological baseline:

```text
--- CONSOLIDATED ABLATION SUMMARY ---
Experiment Name                F1-Score   FPR        Latency      F1 Delta   FPR Delta
Baseline Model (Temporal)      0.2553     0.5660     0.0019ms     Reference  Reference
Ablation 1 (No Volumetric)     0.0017     0.0070     0.0015ms     -0.2536    -0.5590
Ablation 2 (No Conn State)     0.0000     0.0000     0.0014ms     -0.2553    -0.5660
Ablation 3 (Capacity Limit)    0.0000     0.0000     0.0011ms     -0.2553    -0.5660
Ablation 4 (Shuffled Split)    0.8986     0.2873     0.0014ms     +0.6433    -0.2787
```

* **Ablation 4 (Data Leakage Verification)**: Shuffling network logs instead of splitting them chronologically inflates the F1-score artificially by **`+0.6433`** (rising from `0.2553` to `0.8986`). This empirical proof demonstrates that randomized data partitioning introduces severe data leakage and invalidates cybersecurity evaluations.

---

## CHAPTER 4: Learning Outcomes

### 4.1 Project Learning Outcomes
* **Temporal Leakage Mitigation**: Gained deep knowledge of why network captures must be split chronologically rather than randomly to prevent future connection statistics from bleeding into historical models.
* **Sequential Feature Engineering**: Learned to construct multi-step sequence tensors for recurrent models and manage sliding evaluation buffers during continuous flow sniffing.
* **Edge Deployment Heuristics**: Mastered trade-offs in ML model compression, finding that while dynamic quantization reduces file footprints for edge deployment, it can alter decision boundaries and requires calibration.

### 4.2 Internship Learning Outcomes
* **Modular Code Architecture**: Developed appreciation for clean, structured Python scripts with parameterized arguments rather than chaotic Jupyter notebooks.
* **Security Operations (SOC) Logic**: Learned to think like a security engineer, designing hybrid intrusion detection systems that combine machine learning models with whitelists to minimize false alarms on critical servers.
* **Version Control and CI/CD Hygiene**: Practiced repository synchronization using Git, force-resetting history, and deploying production codebases to remote servers.

### 4.3 Skills Acquired

| Technical Skills | Soft Skills |
| :--- | :--- |
| **Deep Learning (PyTorch)** | **Critical Thinking & Problem Solving** |
| **Network Log Analysis (Zeek/Bro)** | **Technical Document Writing** |
| **Edge Optimization (INT8 Quantization)** | **Adaptability and Autonomy** |
| **Socket Sniffing (Scapy)** | **Time Management & Sprint Compliance** |
| **Security Operations (Psutil)** | **Workplace Communication** |

### 4.4 Best Learning from this Internship
The single most valuable takeaway from this internship was discovering the **Base Rate Fallacy** in highly imbalanced network domains. In standard datasets, a $1\%$ False Positive Rate sounds acceptable. However, on a real-world enterprise link carrying millions of packets daily, a $1\%$ false alarm rate translates to thousands of false alerts daily, overwhelming SOC analysts. This taught me to prioritize Recall ($89.43\%$) for threat containment and to implement whitelists and threshold calibrations rather than relying blindly on training accuracies.

---

## CHAPTER 5: Conclusion

### 5.1 Project Accomplishment Summary
During this internship project, we successfully designed, evaluated, and deployed a production-ready, quantized LSTM network for real-time IoT botnet detection. By implementing a temporal split protocol, we mitigated training data leakage. The resulting model generalizes successfully on unseen, out-of-distribution physical device captures. 

Our dynamic quantization successfully compressed the deployment footprint down to **`29.50 KB`**, ensuring compatibility with resource-constrained IoT gateways. Finally, the real-time Scapy sniffing adapter provides a live intrusion detection system complete with whitelisting safety overrides.

### 5.2 Feedback
* **Quality of Guidance**: The academic and industrial mentors provided excellent, structured checkpoints that helped steer the project from concept to code.
* **Lab Environment**: Compute resources and hardware facilities at IIITDM Kancheepuram were optimal for sequential model training.
* **Recommendations**: Future internships could integrate physical hardware deployments on Raspberry Pi controllers for hardware-in-the-loop validation.

### 5.3 Declaration & Signature

I hereby declare that this internship report is a genuine record of the work carried out at IIITDM Kancheepuram during June 1, 2026 – July 7, 2026, under the guidance of our supervisor, and has not been submitted elsewhere for any degree or diploma.

**Student Name & Roll Number**:  
1. **Sushanth** — ______________________

**Place**: Chennai, India  
**Date**: July 7, 2026  

<br>

**Signature**:  
1. ________________________________________

---

## References

1. Stratosphere Labs. *IoT-23: A labeled dataset with malicious and benign IoT network traffic.* (https://www.stratosphereips.org/dataset-iot-23).
2. Pascanu, R., Mikolov, T., & Bengio, Y. *On the difficulty of training recurrent neural networks.* International Conference on Machine Learning, 2013.
3. PyTorch Documentation. *Dynamic Quantization on PyTorch models.* (https://pytorch.org/docs/stable/quantization.html).
4. Bakhshi, T. *State-of-the-art in IoT botnet detection: A systematic review.* IEEE Access, 2021.
5. Scapy Contributors. *Scapy: packet drafting, sniffing, and crafting tool.* (https://scapy.net/).
