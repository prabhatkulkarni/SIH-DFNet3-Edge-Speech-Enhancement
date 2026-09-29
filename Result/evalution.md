# 📊 Model Evaluation & Technical Evidence

This document records the training, evaluation, ONNX export, and deployment-readiness evidence for the SIH DeepFilterNet3 speech-enhancement system.

---

# 1. Model Overview

The project uses **DeepFilterNet3** as the base speech-enhancement architecture.

The model was initialized from the official pretrained DeepFilterNet3 model and fine-tuned using the project's multilingual speech and diverse noise configuration.

### Final checkpoint

```text
Model:              DeepFilterNet3
Final epoch:        120
Checkpoint:         model_120.ckpt.best
Sample rate:        48 kHz
FFT size:           960
Hop size:           480
```

The final checkpoint was selected by the training/evaluation pipeline at epoch 120.

---

# 2. Training Configuration

The final fine-tuning run used:

```text
Initialization:
Official pretrained DeepFilterNet3

Training:
Fine-tuning on the project dataset

Final epoch:
120
```

The training configuration included multilingual speech and multiple noise sources.

### Speech

- English speech
- Hindi speech

### Noise

- DEMAND
- Drone noise
- ESC-50 selected noise classes
- Firearms audio
- MS-SNSD noise

The objective was to train a general speech-enhancement model exposed to a diverse range of acoustic interference.

---

# 3. Final Test-Set Results

The final model was evaluated using the project's test configuration.

### Results

| Evaluation SNR | SDR (dB) | STOI |
|---:|---:|---:|
| -5 dB | 2.85153 | 0.62233 |
| 0 dB | **2.91873** | **0.69247** |
| 5 dB | 6.47873 | 0.80248 |
| 10 dB | 7.18525 | 0.84029 |
| 20 dB | 9.80311 | 0.92150 |
| 40 dB | 12.50221 | 0.99379 |

### 0-dB condition

The 0-dB test condition produced:

```text
SDR  = +2.91873 dB
STOI = 0.69247
```

These are test-set evaluation metrics and should be distinguished from the separate controlled-sample SNR experiment described below.

---

# 4. Controlled 0-dB Evaluation

A deterministic Hindi speech sample was mixed with DEMAND background noise to create an approximately **0-dB input SNR** condition.

### Signal chain

```text
Hindi Clean Speech
        +
DEMAND Noise
        ↓
Approximately 0 dB SNR
        ↓
Noisy Speech
        ↓
Fine-tuned DeepFilterNet3
        ↓
Enhanced Speech
```

### Input characteristics

```text
Sample rate       : 48 kHz
Duration          : ≈ 10.64 s
Input SNR         : ≈ 0 dB
Clean RMS         : 0.0404556
Noisy RMS         : 0.0572189
```

---

# 5. Controlled Evaluation Result

The fine-tuned model produced:

```text
Input SNR       : ≈ -0.000002 dB
Output SNR      : 20.408592 dB
SNR improvement : +20.408594 dB
```

Additional output measurements:

```text
Output RMS      : 0.039104547
Output peak     : 0.35752606
```

### Summary

| Metric | Result |
|---|---:|
| Input SNR | ≈ 0 dB |
| Output SNR | **20.4086 dB** |
| SNR improvement | **≈ +20.4086 dB** |
| Output RMS | 0.0391 |
| Output peak | 0.3575 |

> **Important:** The 20.4086 dB result is from one controlled evaluation sample. It is not the overall test-set average or a guaranteed performance value for every noise condition.

---

# 6. Audio Demonstration

The corresponding audio demonstration is available in the [`Demo`](../Demo/) directory.

The demonstration contains:

```text
clean.wav
noisy_0dB.wav
enhanced_finetuned.wav
results.png
```

The intended demonstration flow is:

```text
Clean Speech
     +
DEMAND Noise
     ↓
Noisy Speech
     ↓
Fine-tuned DeepFilterNet3
     ↓
Enhanced Speech
```

This allows the input and output to be inspected both audibly and visually.

---

# 7. ONNX Export

The fine-tuned DeepFilterNet3 neural-network components were exported to ONNX for edge-deployment preparation.

The exported components are:

```text
enc.onnx
erb_dec.onnx
df_dec.onnx
```

### Approximate model sizes

| Component | Size |
|---|---:|
| `enc.onnx` | 1.86 MB |
| `erb_dec.onnx` | 3.14 MB |
| `df_dec.onnx` | 3.19 MB |
| **Total** | **≈ 8.19 MB** |

The ONNX export uses the split component architecture supported by the DeepFilterNet repository.

---

# 8. ONNX Component Validation

Each exported ONNX component was compared against reference outputs from the PyTorch implementation.

## `enc.onnx`

```text
MAE  ≈ 10⁻⁸ to 10⁻⁷ scale
RMSE ≈ 10⁻⁸ to 10⁻⁷ scale
MAX  ≈ 10⁻⁶ scale
```

Representative maximum errors were approximately:

```text
e0    9.984e-07
e1    7.153e-07
e2    1.192e-06
e3    3.338e-06
emb   2.146e-06
c0    5.722e-06
lsnr  5.722e-06
```

## `erb_dec.onnx`

```text
MAE  = 8.142e-08
RMSE = 1.290e-07
MAX  = 1.043e-06
```

## `df_dec.onnx`

```text
MAE  = 2.469e-09
RMSE = 5.399e-09
MAX  = 1.490e-07
```

### Component validation status

```text
enc.onnx       ✅ PASS
erb_dec.onnx   ✅ PASS
df_dec.onnx    ✅ PASS
```

---

# 9. End-to-End ONNX-Backed Validation

After exporting the neural-network components, the ONNX components were integrated into the DeepFilterNet processing pipeline using **ONNX Runtime**.

The surrounding DeepFilterNet feature extraction, signal processing, and synthesis stages were retained for this validation.

### Validation pipeline

```text
Noisy Audio
     ↓
DeepFilterNet Feature Extraction
     ↓
┌─────────────────────────────┐
│        ONNX Runtime         │
│                             │
│  enc.onnx                   │
│  erb_dec.onnx               │
│  df_dec.onnx                │
└──────────────┬──────────────┘
               ↓
       DeepFilter Processing
               ↓
        Audio Synthesis
               ↓
        Enhanced Audio
```

---

# 10. PyTorch vs ONNX Result

The same controlled approximately 0-dB input was processed through the PyTorch and ONNX-backed pipelines.

| Implementation | Output SNR |
|---|---:|
| PyTorch | 20.408592 dB |
| ONNX Runtime | 20.408594 dB |

Difference:

```text
≈ 0.000002 dB
```

The corresponding ONNX-backed output measurements were:

```text
Input SNR       : -0.000002 dB
ONNX output SNR : 20.408594 dB
Improvement     : 20.408596 dB
Output RMS      : 0.039104548
Output peak     : 0.357526064
```

### Validation conclusion

For this controlled evaluation pipeline, the ONNX-backed neural-network components reproduced the PyTorch result extremely closely.

```text
PyTorch
   │
   ▼
20.408592 dB

ONNX Runtime
   │
   ▼
20.408594 dB

Difference
   │
   ▼
≈ 0.000002 dB
```

This provides evidence that the exported ONNX components are suitable for the next edge-deployment stage.

---

# 11. Deployment Readiness

The project has completed the model-training and ONNX-validation stages.

### Current status

| Stage | Status |
|---|:---:|
| Dataset preparation | 🟢 Complete |
| DeepFilterNet3 fine-tuning | 🟢 Complete |
| Controlled evaluation | 🟢 Complete |
| Test-set evaluation | 🟢 Complete |
| ONNX export | 🟢 Complete |
| ONNX component validation | 🟢 Complete |
| End-to-end ONNX validation | 🟢 Complete |
| Audio demonstration | 🟢 Complete |
| Raspberry Pi 5 deployment | 🟡 Planned |
| Raspberry Pi real-time benchmark | 🟡 Pending |
| INT8 optimization | ⚪ Future |

---

# 12. Raspberry Pi 5 Deployment Plan

The final target platform is **Raspberry Pi 5**.

The ONNX pipeline has been prepared for the edge-deployment stage, but the Raspberry Pi 5 hardware itself has not yet been benchmarked.

### Planned architecture

```text
🎤 Microphone
      ↓
Audio Capture
      ↓
48 kHz PCM
      ↓
Feature Extraction
      ↓
ONNX Runtime
      ↓
┌─────────────────┐
│ DeepFilterNet3  │
│ ONNX Components │
└────────┬────────┘
         ↓
DF Processing
         ↓
Audio Synthesis
         ↓
🔊 Speaker
```

---

# 13. Raspberry Pi Benchmark Plan

When Raspberry Pi 5 hardware is available, the following measurements will be collected:

| Metric | Purpose |
|---|---|
| End-to-end latency | Measure processing delay |
| Real-time factor (RTF) | Determine real-time feasibility |
| CPU utilization | Measure processor load |
| RAM usage | Determine memory requirements |
| Streaming stability | Test continuous audio processing |
| Output audio quality | Evaluate practical enhancement |

The real-time factor will be calculated as:

```text
RTF = Processing Time / Audio Duration
```

A measured RTF below `1.0` would indicate that processing is faster than the corresponding audio duration.

> No Raspberry Pi real-time performance claim is made until this is measured on actual Raspberry Pi 5 hardware.

---

# 14. Current Proof-of-Readiness

The project currently provides evidence at multiple levels:

```text
LEVEL 1 — MODEL
     │
     ├── Fine-tuned DeepFilterNet3
     └── Epoch-120 checkpoint
     │
     ▼
LEVEL 2 — AUDIO
     │
     ├── Clean speech
     ├── Noisy speech
     └── Enhanced speech
     │
     ▼
LEVEL 3 — QUANTITATIVE
     │
     ├── Test-set SDR / STOI
     └── Controlled SNR evaluation
     │
     ▼
LEVEL 4 — EDGE MODEL
     │
     ├── enc.onnx
     ├── erb_dec.onnx
     └── df_dec.onnx
     │
     ▼
LEVEL 5 — ONNX VALIDATION
     │
     └── PyTorch ≈ ONNX
     │
     ▼
LEVEL 6 — HARDWARE
     │
     └── Raspberry Pi 5 benchmark
         [NEXT STAGE]
```

---

# 15. Evidence Files

The repository contains the corresponding evidence:

```text
Demo/
├── README.md
├── clean.wav
├── noisy_0dB.wav
├── enhanced_finetuned.wav
└── results.png

Result/
└── evaluation.md

deployment/
└── RASPBERRY_PI_5.md
```

The `Demo/` directory provides the judge-facing audio and visual demonstration.

This `Result/` document provides the quantitative and technical evidence.

The `deployment/` directory describes the planned Raspberry Pi 5 edge-deployment architecture.

---

# 16. Limitations and Scope

The following points are intentionally kept explicit:

1. Raspberry Pi 5 hardware has not yet been used for a performance benchmark.
2. Raspberry Pi latency has not yet been measured.
3. Raspberry Pi CPU and RAM utilization have not yet been measured.
4. Real-time operation on Raspberry Pi 5 has not yet been experimentally verified.
5. INT8 quantization has not yet been performed.
6. The 20.4086 dB SNR result is from one controlled evaluation sample.
7. Test-set SDR/STOI and controlled-sample SNR are different evaluation measurements and should not be treated as the same metric.

---

# 17. Development Roadmap

```text
                 ✅
       DeepFilterNet3 Fine-Tuning
                 │
                 ▼
                 ✅
          Model Evaluation
                 │
                 ▼
                 ✅
             ONNX Export
                 │
                 ▼
                 ✅
          ONNX Validation
                 │
                 ▼
                 🟡
       Raspberry Pi 5 Deployment
                 │
                 ▼
                 🟡
       Real-Time Benchmarking
                 │
                 ▼
                 ⚪
          FP32 Optimization
                 │
                 ▼
                 ⚪
         INT8 Quantization
                 │
                 ▼
                 ⚪
       Optimized Edge System
```

---

## Final Status

> **The neural speech-enhancement model has been fine-tuned, quantitatively evaluated, exported to ONNX, and validated against the PyTorch implementation. The next stage is hardware validation and real-time benchmarking on Raspberry Pi 5.**
