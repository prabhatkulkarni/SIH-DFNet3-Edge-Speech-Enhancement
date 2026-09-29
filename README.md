# 🎙️ SIH-DFNet3 — Edge Speech Enhancement

### Fine-tuned DeepFilterNet3 for Hindi Speech Enhancement with ONNX-based Edge Deployment Targeting Raspberry Pi 5

A speech enhancement system designed to reduce background noise from human speech while preserving speech quality.

The current model has been fine-tuned for **Hindi speech** and validated through controlled experiments and ONNX Runtime.

The project is designed to be extended toward **multilingual speech enhancement** using the prepared English and Hindi speech datasets together with diverse real-world noise datasets.

---

## 🚀 Project Overview

Background noise can make speech difficult to understand in environments such as:

- Vehicles
- Outdoor locations
- Crowded environments
- Electronic environments
- Industrial environments
- Drone environments
- Emergency situations
- Low-SNR communication

This project uses **DeepFilterNet3**, a neural speech enhancement architecture, and adapts it using our speech and noise datasets.

The trained model is then exported into **ONNX components** for future edge deployment.

### Current pipeline

```text
Speech
  +
Background Noise
       │
       ▼
  Noisy Speech
       │
       ▼
Fine-tuned DeepFilterNet3
       │
       ▼
Enhanced Speech
       │
       ▼
     ONNX
       │
       ▼
Future Edge Deployment
       │
       ▼
 Raspberry Pi 5
```

---

# 🎯 Current Goal

The current experimentally validated model focuses on:

> **Hindi speech enhancement under different noise conditions.**

The model has been:

- Fine-tuned using Hindi speech data
- Evaluated using multiple noise conditions
- Tested using controlled noisy speech
- Exported to ONNX
- Numerically validated against PyTorch
- Integrated into an ONNX-backed inference pipeline

---

# 🌐 Multilingual Expansion

Although the current trained model is Hindi-focused, the project has been designed with a broader multilingual goal.

The prepared datasets include:

```text
English Speech
       +
Hindi Speech
       +
Multiple Noise Sources
       ↓
Multilingual Speech Enhancement
```

The available speech and noise datasets allow the project to be extended beyond Hindi.

### Planned multilingual pipeline

```text
English Speech ─────┐
                    │
Hindi Speech ───────┤
                    │
Future Languages ───┤
                    ▼
             Multilingual Training
                    │
                    ▼
              DeepFilterNet3
                    │
                    ▼
                 ONNX
                    │
                    ▼
              Raspberry Pi 5
```

The goal is to improve robustness across:

- Languages
- Accents
- Speakers
- Recording conditions
- Noise environments
- Signal-to-noise ratios

> **Current status:** The Hindi model is the experimentally validated model. Multilingual training is a planned expansion using the prepared datasets.

---

# 📊 Dataset Strategy

The project combines clean speech datasets with diverse noise datasets.

## Speech

### English

```text
~88,000 speech recordings
~110 speakers
16 kHz source audio
Speaker-disjoint train/validation/test split
```

### Hindi

```text
600 original recordings
~100 speakers
Training + testing data
Original dataset contains heterogeneous audio formats
```

The Hindi dataset was cleaned and converted into a consistent format before training.

---

# 🔊 Noise Sources

The project uses multiple noise sources to expose the model to different real-world conditions.

```text
DEMAND
   │
   ├── Environmental noise
   │
Drone Noise
   │
   ├── Drone + background noise
   │
ESC-50
   │
   ├── Vehicle
   ├── Engine
   ├── Wind
   ├── Rain
   ├── Siren
   └── Electronic / household sounds
   │
Firearms
   │
   └── Firearm acoustic events
   │
MS-SNSD
   │
   └── Complex synthetic / real-world noise
```

This provides a diverse noise environment for speech enhancement experiments.

---

# 🧠 Model

The project uses:

**DeepFilterNet3**

DeepFilterNet3 performs speech enhancement using:

- Spectral features
- ERB-band features
- Neural masking
- Deep filtering
- Time-frequency processing

The model operates at:

```text
Sample Rate : 48 kHz
FFT Size    : 960
Hop Size    : 480
ERB Bands   : 32
DF Bins     : 96
DF Order    : 5
```

---

# 🏋️ Training

The model was fine-tuned from the official pretrained DeepFilterNet3 model.

### Training configuration

```text
Model          : DeepFilterNet3
Initialization : Official pretrained model
Final Epoch    : 120
Optimizer      : AdamW
Learning Rate  : 5e-4
Batch Size     : 16
```

Training was performed using **GPU-enabled Google Colab**.

The training and evaluation pipeline was designed to keep speech and noise data separated across the appropriate splits.

---

# 📈 Evaluation

A controlled Hindi speech sample was mixed with DEMAND environmental noise at approximately:

```text
Input SNR ≈ 0 dB
```

The fine-tuned model produced:

```text
Output SNR ≈ 20.4086 dB
```

### Controlled sample

| Measurement | Result |
|---|---:|
| Input SNR | ≈ 0 dB |
| Output SNR | **20.4086 dB** |
| SNR Improvement | **≈ +20.4086 dB** |

> **Important:** This 20.4086 dB result is from one controlled evaluation sample. It is not the overall test-set average.

---

# 🧪 Test-Set Evaluation

The final model was also evaluated using the project test configuration.

For the approximately 0 dB test condition:

```text
SDR : 2.91873 dB
STOI: 0.69247
```

The test-set evaluation and the controlled-sample SNR experiment are different measurements and should not be directly treated as the same metric.

---

# 🔄 PyTorch → ONNX

For edge deployment, the learned DeepFilterNet3 components were exported to ONNX.

The exported components are:

```text
enc.onnx
erb_dec.onnx
df_dec.onnx
```

The ONNX model package also contains:

```text
config.ini
version.txt
```

### Model structure

```text
Input Features
      │
      ▼
┌──────────────┐
│   enc.onnx   │
└──────┬───────┘
       │
       ▼
┌────────────────┐
│  erb_dec.onnx  │
└──────┬─────────┘
       │
       ▼
 DeepFilter Processing
       │
       ▼
┌──────────────┐
│  df_dec.onnx │
└──────┬───────┘
       │
       ▼
 Enhanced Spectrum
       │
       ▼
 Audio Synthesis
```

---

# ✅ ONNX Validation

Each ONNX component was compared against its PyTorch reference output.

| Component | Validation |
|---|:---:|
| `enc.onnx` | ✅ PASS |
| `erb_dec.onnx` | ✅ PASS |
| `df_dec.onnx` | ✅ PASS |

Example numerical errors:

| Component | MAE | RMSE |
|---|---:|---:|
| `enc.onnx` | `2.827e-08`* | `7.447e-08`* |
| `erb_dec.onnx` | `8.142e-08` | `1.290e-07` |
| `df_dec.onnx` | `2.469e-09` | `5.399e-09` |

\* Representative encoder output comparison.

---

# 🔬 End-to-End ONNX Validation

The exported ONNX components were integrated into the DeepFilterNet processing pipeline using **ONNX Runtime**.

The same controlled input was processed using both implementations.

| Implementation | Output SNR |
|---|---:|
| PyTorch | `20.408592 dB` |
| ONNX Runtime | `20.408594 dB` |

Difference:

```text
≈ 0.000002 dB
```

This demonstrates that the exported ONNX neural-network components reproduce the PyTorch result extremely closely for the controlled evaluation pipeline.

---

# 💻 Development & Deployment Environment

Different stages of the project use different environments.

| Environment | Purpose |
|---|---|
| **Google Colab + GPU** | Training, evaluation and ONNX export |
| **GitHub** | Code, ONNX model files, results and documentation |
| **Local / Codespaces** | Repository development and lightweight testing |
| **Raspberry Pi 5** | Intended edge deployment and hardware benchmarking |

### Development flow

```text
Datasets
   │
   ▼
Google Colab + GPU
   │
   ├── Data preparation
   ├── Model training
   ├── Evaluation
   └── ONNX export
   │
   ▼
GitHub Repository
   │
   ├── ONNX Models
   ├── Inference Code
   ├── Results
   └── Documentation
   │
   ▼
Raspberry Pi 5
   │
   ├── Audio Input
   ├── ONNX Runtime
   ├── Speech Enhancement
   └── Audio Output
```

> Model training and ONNX export require significantly more computational resources than normal repository development. GPU-enabled Google Colab is therefore used for the heavy computational stages.

---

# 🍓 Raspberry Pi 5 Deployment

The final target platform is:

**Raspberry Pi 5**

The intended system is:

```text
🎤 Microphone
      │
      ▼
Audio Capture
      │
      ▼
48 kHz PCM
      │
      ▼
Feature Extraction
      │
      ▼
┌───────────────────────┐
│     ONNX Runtime      │
│                       │
│  enc.onnx             │
│  erb_dec.onnx         │
│  df_dec.onnx          │
└───────────┬───────────┘
            │
            ▼
    DeepFilter Processing
            │
            ▼
      Audio Synthesis
            │
            ▼
       🔊 Speaker
```

The Raspberry Pi will eventually perform the speech enhancement locally, reducing the need for cloud processing.

---

# ⚡ Real-Time Deployment Plan

After Raspberry Pi 5 hardware is available, the system will be tested for:

- End-to-end latency
- Processing time
- Real-time factor
- CPU utilization
- RAM usage
- Continuous streaming stability
- Audio quality

### Real-Time Factor

```text
RTF = Processing Time / Audio Duration
```

```text
RTF < 1.0
    ↓
Processing is faster than the audio duration
```

No Raspberry Pi real-time performance number is claimed until it is measured on actual Raspberry Pi 5 hardware.

---

# 💰 Future Expansion With More Resources

The current implementation is a working foundation.

With additional **computational resources, time, budget and data**, the project can be expanded significantly.

### Planned improvements

```text
Current Hindi Model
       │
       ▼
More Training Data
       │
       ▼
Multilingual Training
       │
       ▼
More Noise Conditions
       │
       ▼
More Model Experiments
       │
       ▼
Hyperparameter Optimization
       │
       ▼
Model Optimization
       │
       ▼
Quantization
       │
       ▼
Raspberry Pi Optimization
       │
       ▼
Real-Time Edge System
```

Possible improvements include:

- More languages
- More accents
- More speakers
- Larger speech datasets
- More real-world noise
- More SNR conditions
- Longer training
- Hyperparameter optimization
- Architecture experiments
- FP32 optimization
- INT8 quantization
- Streaming optimization
- Raspberry Pi performance optimization

---

# 📁 Repository Structure

```text
SIH-DFNet3-Edge-Speech-Enhancement/
│
├── Demo/
│   ├── README.md
│   ├── clean.wav
│   ├── noisy_0dB.wav
│   ├── enhanced_finetuned.wav
│   ├── enhanced_onnx_hybrid.wav
│   └── results.png
│
├── Models/
│   └── dfnet3_hindi_onnx/
│       ├── config.ini
│       ├── enc.onnx
│       ├── erb_dec.onnx
│       ├── df_dec.onnx
│       └── version.txt
│
├── Result/
│   └── evaluation.md
│
├── inference/
│   ├── README.md
│   ├── enhance_onnx.py
│   └── requirements.txt
│
├── deployment/
│   └── RASPBERRY_PI_5.md
│
├── .gitignore
└── README.md
```

---

# 🔧 Inference

The repository contains an ONNX-backed inference implementation.

```text
inference/
├── enhance_onnx.py
├── requirements.txt
└── README.md
```

The inference pipeline uses:

```text
Input WAV
    ↓
DeepFilterNet Feature Extraction
    ↓
ONNX Runtime
    ↓
enc.onnx
erb_dec.onnx
df_dec.onnx
    ↓
DeepFilter Processing
    ↓
Audio Synthesis
    ↓
Enhanced WAV
```

Detailed instructions are available in:

```text
inference/README.md
```

---

# 📚 DeepFilterNet Source

This project is based on the DeepFilterNet implementation.

Pinned source commit used during development:

```text
d375b2d8309e0935d165700c91da9de862a99c31
```

The repository's inference implementation retains the required DeepFilterNet processing stages around the exported ONNX components.

---

# 🧪 Current Project Status

```text
Dataset Preparation              ✅
Hindi Speech Preparation         ✅
Noise Dataset Preparation        ✅
DeepFilterNet3 Fine-Tuning       ✅
Controlled Evaluation            ✅
Test-Set Evaluation              ✅
ONNX Export                      ✅
ONNX Component Validation        ✅
End-to-End ONNX Validation       ✅
GitHub Model Artifacts           ✅
GitHub Inference Code            ✅
Deployment Documentation         ✅
Multilingual Expansion           🟡 Planned
Raspberry Pi 5 Deployment        🟡 Planned
Real-Time Benchmark              🟡 Pending
INT8 Quantization                ⚪ Future
```

---

# 🏆 Proof of Readiness

The project currently provides evidence at multiple levels:

```text
LEVEL 1
Dataset Preparation
       │
       ▼
LEVEL 2
DeepFilterNet3 Fine-Tuning
       │
       ▼
LEVEL 3
Audio + Quantitative Evaluation
       │
       ▼
LEVEL 4
ONNX Export
       │
       ▼
LEVEL 5
ONNX Component Validation
       │
       ▼
LEVEL 6
End-to-End ONNX Validation
       │
       ▼
LEVEL 7
GitHub Inference Package
       │
       ▼
LEVEL 8
Raspberry Pi 5 Deployment
       │
       ▼
LEVEL 9
Real-Time Hardware Benchmark
```

The first seven stages have been completed.

The remaining stages require Raspberry Pi 5 hardware testing.

---

# 🛣️ Roadmap

```text
                         STATUS
                           │
                           ▼
              ┌─────────────────────┐
              │ Dataset Preparation │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Hindi Model Training│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Evaluation    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ ONNX Export         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ ONNX Validation     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ GitHub Inference    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Multilingual Model  │
              │ Expansion           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Raspberry Pi 5      │
              │ Deployment          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Real-Time Benchmark │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Optimization +      │
              │ Quantization        │
              └─────────────────────┘
```

---

# ⚠️ Limitations

The following limitations are intentionally stated clearly:

1. The currently validated trained model is Hindi-focused.
2. Multilingual training is a planned expansion, not yet claimed as completed.
3. Raspberry Pi 5 hardware has not yet been benchmarked.
4. Raspberry Pi latency has not yet been measured.
5. Raspberry Pi CPU and RAM usage have not yet been measured.
6. Real-time microphone-to-speaker operation has not yet been experimentally verified on Raspberry Pi 5.
7. INT8 quantization has not yet been performed.
8. The 20.4086 dB SNR result is from one controlled evaluation sample.
9. Controlled-sample SNR and test-set SDR/STOI are different measurements.
10. The current ONNX implementation is an ONNX-backed hybrid pipeline rather than a single completely monolithic ONNX graph.

---

# 🎯 Final Vision

The long-term goal is to develop a **multilingual, low-latency, edge-based speech enhancement system** that can operate locally on resource-constrained hardware.

```text
        MULTILINGUAL SPEECH
                 │
                 ▼
        Diverse Noise Conditions
                 │
                 ▼
          DeepFilterNet3
                 │
                 ▼
               ONNX
                 │
                 ▼
          Raspberry Pi 5
                 │
                 ▼
       Real-Time Enhancement
                 │
                 ▼
          Clearer Speech
```

The current Hindi model and ONNX pipeline form the foundation for this larger system.

---

## 👨‍💻 Project

**SIH-DFNet3 — Edge Speech Enhancement**

Built as part of **Smart India Hackathon (SIH)**.

### Core Technologies

```text
Python
PyTorch
DeepFilterNet3
ONNX
ONNX Runtime
NumPy
SciPy
libDF
Google Colab
Raspberry Pi 5
```

---

## 📌 Final Status

> **A Hindi-focused DeepFilterNet3 speech enhancement model has been fine-tuned, evaluated, exported to ONNX, and validated against the PyTorch implementation. The prepared multilingual speech and noise datasets provide a foundation for future multilingual training. The next major stages are expanded training, Raspberry Pi 5 deployment, real-time benchmarking, and edge optimization.**
