# Raspberry Pi 5 Deployment Plan

## 1. Objective

The target deployment platform is **Raspberry Pi 5**.

The project has completed the model-training, evaluation, ONNX export,
and ONNX validation stages. The next stage is to move the validated
inference pipeline to Raspberry Pi 5 and measure its real-time
performance.

> **Current status:** The Raspberry Pi 5 hardware has not yet been
> benchmarked. Therefore, this document describes the deployment
> architecture and validation procedure rather than claiming Raspberry
> Pi performance.

------------------------------------------------------------------------

## 2. Current Readiness

  Component                               Status
  ----------------------------------- ---------------
  DeepFilterNet3 fine-tuning            ✅ Complete
  Hindi speech evaluation               ✅ Complete
  Controlled 0-dB evaluation            ✅ Complete
  ONNX export                           ✅ Complete
  `enc.onnx` validation                   ✅ PASS
  `erb_dec.onnx` validation               ✅ PASS
  `df_dec.onnx` validation                ✅ PASS
  End-to-end ONNX-backed validation       ✅ PASS
  GitHub inference implementation       ✅ Complete
  Raspberry Pi 5 deployment            🟡 Next stage
  Raspberry Pi 5 benchmark              🟡 Pending
  INT8 optimization                      ⚪ Future

------------------------------------------------------------------------

## 3. Target Edge Architecture

``` text
                    Raspberry Pi 5
┌─────────────────────────────────────────────────────┐
│                                                     │
│  🎤 USB / I2S Microphone                            │
│              │                                      │
│              ▼                                      │
│       Audio Capture                                 │
│              │                                      │
│              ▼                                      │
│       48 kHz PCM Audio                              │
│              │                                      │
│              ▼                                      │
│   DeepFilterNet Feature Extraction                  │
│              │                                      │
│              ▼                                      │
│        ┌───────────────────┐                        │
│        │   ONNX Runtime    │                        │
│        │                   │                        │
│        │   enc.onnx        │                        │
│        │   erb_dec.onnx    │                        │
│        │   df_dec.onnx     │                        │
│        └─────────┬─────────┘                        │
│                  │                                  │
│                  ▼                                  │
│        DF Processing + Mask                         │
│                  │                                  │
│                  ▼                                  │
│          Audio Synthesis                            │
│                  │                                  │
│                  ▼                                  │
│       🔊 Speaker / Application                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

The current inference implementation uses ONNX Runtime for the exported
learned components while retaining the surrounding DeepFilterNet
feature-extraction, tensor-processing, and synthesis stages.

------------------------------------------------------------------------

## 4. ONNX Model Package

The repository contains the exported model components under:

``` text
Models/
└── dfnet3_hindi_onnx/
    ├── config.ini
    ├── enc.onnx
    ├── erb_dec.onnx
    ├── df_dec.onnx
    └── version.txt
```

### Model configuration

``` text
Sample rate : 48 kHz
FFT size    : 960
Hop size    : 480
ERB bands   : 32
DF bins     : 96
DF order    : 5
```

The ONNX export was generated from the fine-tuned DeepFilterNet3
checkpoint at epoch 120.

------------------------------------------------------------------------

## 5. ONNX Validation Evidence

The three exported neural-network components were validated against
their PyTorch reference outputs.

  Component                        MAE          RMSE   Maximum Error  Status
  ---------------- ------------------- ------------- --------------- --------
  `enc.onnx`         `2.827e-08` (emb)   `7.447e-08`     `2.146e-06`    ✅
  `erb_dec.onnx`           `8.142e-08`   `1.290e-07`     `1.043e-06`    ✅
  `df_dec.onnx`            `2.469e-09`   `5.399e-09`     `1.490e-07`    ✅

These are component-level numerical comparisons against reference
vectors.

------------------------------------------------------------------------

## 6. End-to-End ONNX Validation

A controlled approximately 0-dB Hindi speech/noise sample was processed
through:

``` text
Input WAV
   ↓
DeepFilterNet feature extraction
   ↓
ONNX Runtime
   ├── enc.onnx
   ├── erb_dec.onnx
   └── df_dec.onnx
   ↓
DeepFilterNet processing
   ↓
Audio synthesis
   ↓
Enhanced WAV
```

### PyTorch vs ONNX-backed result

  Implementation         Output SNR
  ---------------- ----------------
  PyTorch            `20.408592 dB`
  ONNX Runtime       `20.408594 dB`

Difference:

``` text
≈ 0.000002 dB
```

For this controlled sample:

``` text
Input SNR       : -0.000002 dB
ONNX output SNR : 20.408594 dB
SNR improvement : 20.408596 dB
```

> **Important:** The 20.4086 dB result is from one controlled evaluation
> sample. It is not a Raspberry Pi benchmark and should not be presented
> as an overall test-set average.

------------------------------------------------------------------------

## 7. Raspberry Pi Software Stack

The planned software stack is:

``` text
Raspberry Pi OS / Linux
        │
        ├── Python
        ├── ONNX Runtime
        ├── NumPy
        ├── SoundFile / audio I/O
        ├── DeepFilterNet/libDF
        └── Project inference code
```

The exact dependency versions will be pinned after the first Raspberry
Pi deployment test.

The inference implementation is available in:

``` text
inference/
├── README.md
├── enhance_onnx.py
└── requirements.txt
```

------------------------------------------------------------------------

## 8. Deployment Procedure

### Step 1 --- Prepare Raspberry Pi 5

Install the required operating system and system audio dependencies.

### Step 2 --- Clone the repository

``` bash
git clone https://github.com/prabhatkulkarni/SIH-DFNet3-Edge-Speech-Enhancement.git
cd SIH-DFNet3-Edge-Speech-Enhancement
```

### Step 3 --- Install Python dependencies

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r inference/requirements.txt
```

### Step 4 --- Install the pinned DeepFilterNet source

The current inference implementation depends on the DeepFilterNet source
used during validation.

Pinned upstream commit:

``` text
d375b2d8309e0935d165700c91da9de862a99c31
```

Set:

``` bash
export DEEPFILTERNET_ROOT=/path/to/DeepFilterNet/DeepFilterNet
```

### Step 5 --- Run offline inference

``` bash
python inference/enhance_onnx.py \
    --model-dir Models/dfnet3_hindi_onnx \
    input.wav \
    enhanced.wav
```

### Step 6 --- Verify output

Check:

-   output sample rate
-   output duration
-   audio integrity
-   enhancement quality
-   processing time

------------------------------------------------------------------------

## 9. Streaming Deployment

After offline inference is verified, the next implementation stage is
streaming audio.

Planned pipeline:

``` text
Microphone
    ↓
Audio Buffer
    ↓
Fixed-size Processing Chunks
    ↓
Feature Extraction
    ↓
ONNX Runtime
    ↓
DeepFilter Processing
    ↓
Synthesis
    ↓
Output Buffer
    ↓
Speaker
```

The streaming implementation should maintain persistent model state
where required rather than reinitializing the model for every audio
chunk.

------------------------------------------------------------------------

## 10. Raspberry Pi 5 Benchmark Plan

When Raspberry Pi 5 hardware is available, the following measurements
will be recorded.

  Metric                   Measurement
  ------------------------ ----------------------------------
  End-to-end latency       ms
  Inference time           ms
  Real-time factor         ratio
  CPU utilization          \%
  RAM usage                MB
  Audio buffer underruns   count
  Continuous runtime       duration
  Output quality           objective + listening evaluation

### Real-Time Factor

``` text
RTF = Processing Time / Audio Duration
```

Interpretation:

``` text
RTF < 1.0  → processing is faster than audio playback
RTF = 1.0  → processing keeps pace with audio
RTF > 1.0  → processing is slower than real time
```

For streaming, latency and buffer stability will also be measured
because an offline RTF alone does not establish usable interactive
performance.

------------------------------------------------------------------------

## 11. Benchmark Methodology

The benchmark should use the same input audio for each configuration.

### Offline benchmark

``` text
1. Load fixed WAV
2. Warm up runtime
3. Process audio
4. Record processing time
5. Repeat multiple times
6. Report median and percentile latency
```

### Streaming benchmark

``` text
1. Capture microphone audio
2. Process fixed-size chunks
3. Measure per-chunk processing time
4. Monitor buffer underruns
5. Monitor CPU/RAM
6. Run continuously
7. Record stability
```

The first benchmark should establish a baseline before any optimization.

------------------------------------------------------------------------

## 12. Future Optimization

If Raspberry Pi performance is insufficient, optimization can proceed in
stages:

``` text
Baseline ONNX inference
        ↓
Measure bottlenecks
        ↓
Reduce unnecessary copies
        ↓
Optimize audio buffering
        ↓
Optimize ONNX Runtime execution
        ↓
Threading / CPU tuning
        ↓
Model quantization
        ↓
INT8 evaluation
        ↓
Re-benchmark
```

INT8 quantization is a **future optimization step** and has not been
validated for this model yet.

------------------------------------------------------------------------

## 13. Current Proof-of-Readiness

The project provides evidence at multiple levels:

``` text
LEVEL 1
Fine-tuned DeepFilterNet3
        ↓
LEVEL 2
Controlled + test-set evaluation
        ↓
LEVEL 3
ONNX export
        ↓
LEVEL 4
Component numerical validation
        ↓
LEVEL 5
End-to-end ONNX-backed validation
        ↓
LEVEL 6
Reproducible GitHub inference package
        ↓
LEVEL 7
Raspberry Pi 5 hardware benchmark
        ↓
NEXT STAGE
```

This separates **software/model readiness** from **hardware performance
validation**.

------------------------------------------------------------------------

## 14. Evidence Available in This Repository

``` text
Demo/
├── README.md
├── clean.wav
├── noisy_0dB.wav
├── enhanced_finetuned.wav
├── enhanced_onnx_hybrid.wav
└── results.png

Models/
└── dfnet3_hindi_onnx/
    ├── config.ini
    ├── enc.onnx
    ├── erb_dec.onnx
    ├── df_dec.onnx
    └── version.txt

Result/
└── evaluation.md

inference/
├── README.md
├── enhance_onnx.py
└── requirements.txt

deployment/
└── RASPBERRY_PI_5.md
```

------------------------------------------------------------------------

## 15. Limitations

The following are intentionally explicit:

1.  Raspberry Pi 5 hardware has not yet been benchmarked.
2.  Raspberry Pi latency has not yet been measured.
3.  Raspberry Pi CPU utilization has not yet been measured.
4.  Raspberry Pi RAM usage has not yet been measured.
5.  Real-time microphone-to-speaker operation on Raspberry Pi 5 has not
    yet been experimentally verified.
6.  INT8 quantization has not yet been performed.
7.  The 20.4086 dB SNR result comes from one controlled sample.
8.  Controlled-sample SNR and test-set SDR/STOI are different
    measurements.
9.  The current inference implementation is an ONNX-backed hybrid
    pipeline rather than a completely monolithic ONNX graph.

------------------------------------------------------------------------

## 16. Development Roadmap

``` text
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
                 ✅
       GitHub Inference Package
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

------------------------------------------------------------------------

## Final Status

> **The fine-tuned DeepFilterNet3 model has been evaluated, exported
> into ONNX components, numerically validated against PyTorch, and
> integrated into a reproducible ONNX-backed inference pipeline.
> Raspberry Pi 5 deployment and real-time benchmarking are the next
> hardware-validation stages.**
