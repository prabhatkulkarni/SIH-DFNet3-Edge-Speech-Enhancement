# SIH-DFNet3 Edge Speech Enhancement

> **Fine-tuned DeepFilterNet3 for multilingual speech enhancement with ONNX-based edge deployment targeting Raspberry Pi 5.**

---

## 🚀 Project Overview

This project develops an AI-based speech enhancement system designed to suppress diverse environmental, mechanical, and impulsive noise while preserving speech intelligibility.

The system is based on **DeepFilterNet3**, fine-tuned using multilingual speech and a diverse noise corpus. The trained model has subsequently been exported into **ONNX components** and validated using **ONNX Runtime**.

The final deployment target is a **Raspberry Pi 5**, where the system is intended to perform edge-based speech enhancement.

### Target Application

```text
🎤 Microphone
      │
      ▼
 Noisy Speech
      │
      ▼
 DeepFilterNet3
      │
      ▼
 ONNX Runtime
      │
      ▼
 Enhanced Speech
      │
      ▼
 🔊 Speaker
```

---

# 🧠 System Architecture

```text
                         TRAINING
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
      English            Hindi          Noise Corpus
       Speech            Speech         ┌─────────────┐
                                        │ DEMAND      │
                                        │ Drone       │
                                        │ ESC-50      │
                                        │ Firearms    │
                                        │ MS-SNSD     │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        Noisy Speech
                                               │
                                               ▼
                                  Fine-tuned DeepFilterNet3
                                               │
                                               ▼
                                           Epoch 120
                                               │
                                               ▼
                                         ONNX Export
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                     enc.onnx             erb_dec.onnx          df_dec.onnx
                         │                     │                     │
                         └─────────────────────┼─────────────────────┘
                                               │
                                               ▼
                                         ONNX Runtime
                                               │
                                               ▼
                                      Edge Deployment
                                               │
                                               ▼
                                        Raspberry Pi 5
                                               │
                                               ▼
                                      🎤 → DFNet3 → 🔊
```

---

# 🎯 Objectives

The project aims to:

- Enhance speech in noisy environments.
- Support both **English and Hindi speech**.
- Train using diverse environmental and mechanical noise.
- Preserve speech intelligibility while suppressing unwanted noise.
- Convert the trained neural network into an **ONNX-based edge inference pipeline**.
- Prepare the system for deployment on resource-constrained edge hardware.
- Evaluate the feasibility of real-time speech enhancement on a **Raspberry Pi 5**.

---

# 📚 Dataset

The training pipeline uses multilingual speech together with diverse environmental, mechanical, and impulsive noise sources.

## Speech Sources

### English

English speech with various accents was used as the primary English speech corpus.

### Hindi

A Hindi speech dataset was used for multilingual speech enhancement.

## Noise Sources

The project combines multiple noise sources to expose the model to different acoustic conditions:

- **DEMAND** — environmental/background noise
- **Drone Noise Dataset** — drone and related noise
- **ESC-50** — selected environmental and mechanical sounds
- **Firearms Audio Dataset** — firearm noise
- **MS-SNSD** — complex noise

Representative noise types include:

| Noise type | Example source |
|---|---|
| 🔫 Firearm | Firearms dataset |
| 🚁 Drone | Drone dataset |
| 🚗 Vehicle / engine | ESC-50 |
| 🚨 Siren | ESC-50 |
| ✈️ Airplane | ESC-50 |
| 🌬️ Wind | ESC-50 |
| 🌧️ Rain | ESC-50 |
| ⛈️ Thunderstorm | ESC-50 |
| 🧹 Vacuum cleaner | ESC-50 |
| ⌨️ Keyboard typing | ESC-50 |
| 🧺 Washing machine | ESC-50 |

The model is trained as a **general speech-enhancement system**, rather than as separate noise-specific classifiers.

---

# 🧠 Model

## Base Model

**DeepFilterNet3**

DeepFilterNet3 is a neural speech-enhancement architecture combining spectral processing with a learned deep-filtering stage.

## Fine-Tuning

The model was fine-tuned using the project's multilingual speech and diverse noise corpus.

### Final checkpoint

```text
Model:          DeepFilterNet3
Final Epoch:    120
Sample Rate:    48 kHz
FFT Size:       960
Hop Size:       480
```

The final trained checkpoint was:

```text
model_120.ckpt.best
```

---

# 🔬 Training Pipeline

```text
Clean Speech
     │
     ├───────────────┐
     │               │
     │           Noise Sample
     │               │
     └───────┬───────┘
             ▼
        Noise Mixing
             │
             ▼
        Noisy Speech
             │
             ▼
      DeepFilterNet3
             │
             ▼
      Enhanced Speech
             │
             ▼
          Losses
             │
             ▼
       Model Update
```

The training pipeline exposes the model to different speech/noise combinations and SNR conditions.

---

# 📈 Quantitative Validation

## Controlled 0-dB Evaluation

A controlled Hindi speech + DEMAND noise sample was created at approximately **0 dB input SNR**.

```text
Clean Speech
     +
DEMAND Noise
     │
     ▼
  ~0 dB SNR
     │
     ▼
Fine-tuned DeepFilterNet3
     │
     ▼
Enhanced Speech
```

### Result

| Metric | Result |
|---|---:|
| Input SNR | ≈ 0 dB |
| Output SNR | **20.4086 dB** |
| SNR Improvement | **≈ +20.4086 dB** |

> **Evaluation note:** The 20.4086 dB result is from one controlled evaluation sample. It should not be interpreted as the overall test-set average performance of the model.

---

# 📊 Test-Set Evaluation

The final model was also evaluated using the project's test configuration.

At the **0-dB evaluation condition**:

| Metric | Result |
|---|---:|
| SDR | **+2.9187 dB** |
| STOI | **0.6925** |

These metrics represent a different evaluation setup from the controlled-sample SNR measurement above.

---

# ⚙️ ONNX Edge Deployment

The fine-tuned DeepFilterNet3 model was exported into ONNX neural-network components.

The resulting deployment components are:

```text
enc.onnx
erb_dec.onnx
df_dec.onnx
```

These components are used as part of the neural-network inference pipeline while the surrounding DeepFilterNet signal-processing stages remain part of the overall enhancement system.

## 📦 ONNX Model Size

| Component | Approx. Size |
|---|---:|
| `enc.onnx` | 1.86 MB |
| `erb_dec.onnx` | 3.14 MB |
| `df_dec.onnx` | 3.19 MB |
| **Total** | **≈ 8.19 MB** |

---

# ✅ ONNX Validation

The exported ONNX components were tested using **ONNX Runtime** and compared against the original PyTorch implementation.

```text
                         PyTorch
                            │
                            ▼
                       Fine-tuned
                       DeepFilterNet3
                            │
                            ▼
                       ONNX Export
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           enc.onnx     erb_dec.onnx   df_dec.onnx
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     ONNX Runtime
                            │
                            ▼
                     Enhanced Audio
```

### Validation Result

On the controlled evaluation sample:

```text
PyTorch output SNR : 20.408592 dB
ONNX output SNR    : 20.408594 dB
```

Approximate difference:

```text
≈ 0.000002 dB
```

This demonstrates that the exported ONNX-backed neural components reproduce the PyTorch result extremely closely for the controlled evaluation pipeline.

---

# 🧪 ONNX Component Validation

The individual exported components were also numerically compared against reference outputs.

### `enc.onnx`

```text
MAE / RMSE / MAX error
within approximately 10⁻⁶ scale
```

### `erb_dec.onnx`

```text
MAE / RMSE / MAX error
within approximately 10⁻⁶ scale
```

### `df_dec.onnx`

```text
MAE / RMSE / MAX error
within approximately 10⁻⁷ scale
```

These component-level checks provide additional evidence that the exported neural-network components are functioning consistently with the PyTorch implementation.

---

# 🎧 Speech Enhancement Demonstration

The demonstration pipeline is:

```text
                INPUT
                  │
                  ▼
          Noisy Speech
                  │
                  ▼
          DeepFilterNet3
                  │
                  ▼
          Enhanced Speech
                  │
                  ▼
              OUTPUT
```

Example scenarios include:

```text
🎤 Speech + 🔫 Firearm noise
          ↓
       DFNet3
          ↓
    Enhanced Speech


🎤 Speech + 🚁 Drone noise
          ↓
       DFNet3
          ↓
    Enhanced Speech


🎤 Speech + 🚗 Vehicle noise
          ↓
       DFNet3
          ↓
    Enhanced Speech


🎤 Speech + 🚨 Siren
          ↓
       DFNet3
          ↓
    Enhanced Speech
```

The purpose of these demonstrations is to show the model's behavior under different noise conditions represented in the training corpus.

---

# 🥧 Raspberry Pi 5 Deployment

## Target Hardware

The final edge deployment target is:

**Raspberry Pi 5**

The Raspberry Pi 5 has not yet been used for the project's hardware benchmark.

Therefore, Raspberry Pi performance metrics are **not claimed yet**.

## Planned Edge Architecture

```text
                    🎤
                Microphone
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
              ONNX Runtime
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       enc.onnx           erb_dec.onnx
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
                 df_dec
                    │
                    ▼
             DF Processing
                    │
                    ▼
             Audio Synthesis
                    │
                    ▼
                    🔊
                 Speaker
```

---

# 📏 Planned Raspberry Pi Benchmark

Once the system is deployed on Raspberry Pi 5, the following measurements will be collected:

| Metric | Purpose |
|---|---|
| End-to-end latency | Measure response delay |
| Real-time factor | Determine real-time feasibility |
| CPU utilization | Measure processor load |
| RAM usage | Determine memory requirements |
| Streaming stability | Test continuous operation |
| Audio quality | Evaluate enhancement quality |

### Real-Time Criterion

The real-time factor will be calculated as:

```text
RTF = Processing Time / Audio Duration
```

A value below `1.0` indicates that the processing time is shorter than the corresponding audio duration.

No Raspberry Pi real-time claim will be made until this is measured on actual hardware.

---

# 🟢 Proof of Readiness

The current project status is:

| Stage | Status | Evidence |
|---|:---:|---|
| Dataset preparation | 🟢 Complete | Multilingual + diverse noise corpus |
| DeepFilterNet3 fine-tuning | 🟢 Complete | Epoch-120 checkpoint |
| Controlled evaluation | 🟢 Complete | 20.4086 dB output SNR |
| Test-set evaluation | 🟢 Complete | SDR / STOI |
| ONNX export | 🟢 Complete | 3 ONNX components |
| ONNX component validation | 🟢 Complete | Numerical comparison |
| End-to-end ONNX validation | 🟢 Complete | PyTorch equivalence |
| Audio demonstration | 🟡 In progress | Demo package |
| Raspberry Pi 5 deployment | 🟡 Planned | Edge architecture prepared |
| Raspberry Pi real-time benchmark | 🟡 Pending | Requires hardware |
| INT8 optimization | ⚪ Future | After FP32 benchmark |

---

# 🗺️ Deployment Roadmap

```text
        MODEL DEVELOPMENT
               │
               ▼
      Fine-tuned DFNet3
               │
               ▼
       Controlled Testing
               │
               ▼
          ONNX Export
               │
               ▼
     ONNX Runtime Validation
               │
               ▼
        ┌──────────────┐
        │ CURRENT      │
        │ MILESTONE    │
        └──────────────┘
               │
               ▼
       Raspberry Pi 5
               │
               ▼
       Audio Streaming
               │
               ▼
      Real-Time Benchmark
               │
               ▼
       FP32 Optimization
               │
               ▼
        INT8 Quantization
               │
               ▼
      Optimized Edge AI
```

---

# 🛠️ Technology Stack

### Machine Learning

- Python
- PyTorch
- DeepFilterNet3

### Signal Processing

- STFT
- ERB feature processing
- Deep filtering
- Audio synthesis

### Edge Deployment

- ONNX
- ONNX Runtime
- Linux
- Raspberry Pi 5

---

# 📂 Project Structure

```text
SIH-DFNet3-Edge-Speech-Enhancement/
│
├── README.md
│
├── docs/
│   ├── architecture.png
│   ├── deployment.png
│   └── results.png
│
├── demo/
│   ├── README.md
│   └── screenshots/
│
├── inference/
│   └── README.md
│
├── deployment/
│   └── RASPBERRY_PI_5.md
│
└── results/
    └── README.md
```

---

# 📦 Deployment Package

The ONNX deployment package contains:

```text
enc.onnx
erb_dec.onnx
df_dec.onnx
config.ini
version.txt
```

The package is intended to provide the neural-network components required for the edge inference pipeline.

---

# 🔗 Resources

### Official DeepFilterNet Repository

[Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

### Raspberry Pi 5

[Raspberry Pi 5 — Official](https://www.raspberrypi.com/products/raspberry-pi-5/)

### Project Resources

- 📊 Model evaluation results — available in this repository
- 🎧 Audio demonstration — being prepared
- 🎥 Technical demonstration video — being prepared
- 📦 ONNX deployment package — being prepared
- 🥧 Raspberry Pi 5 deployment guide — being prepared

---

# ⚠️ Current Limitations

The following limitations are explicitly acknowledged:

1. Raspberry Pi 5 hardware has not yet been benchmarked.
2. Raspberry Pi latency has not yet been measured.
3. Raspberry Pi CPU and RAM usage have not yet been measured.
4. Real-time operation on Raspberry Pi 5 has not yet been experimentally verified.
5. INT8 quantization has not yet been performed.
6. The 20.4086 dB SNR result represents one controlled evaluation sample and is not an overall test-set average.

---

# 🎯 Next Development Stage

```text
ONNX VALIDATION
       │
       ▼
Raspberry Pi 5 Deployment
       │
       ▼
Microphone Input
       │
       ▼
Real-Time ONNX Inference
       │
       ▼
Audio Output
       │
       ▼
Latency / CPU / RAM Measurement
       │
       ▼
FP32 Optimization
       │
       ▼
INT8 Optimization
```

---

# 🏆 Smart India Hackathon

Developed as part of **Smart India Hackathon (SIH)**.

### Project Vision

> **From a trained speech-enhancement model to a validated ONNX pipeline and ultimately to a deployable edge-AI speech enhancement system.**

---

## 👥 Team

**SIH Team Project**

For project details, implementation evidence, demonstrations, and deployment documentation, explore the repository contents.
