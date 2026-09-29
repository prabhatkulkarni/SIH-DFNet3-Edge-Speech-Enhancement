# ONNX Inference

Reference inference application for the fine-tuned DeepFilterNet3 model.

## Pipeline

```text
WAV / microphone audio
        ↓
DeepFilterNet feature extraction
        ↓
      ONNX Runtime
   ┌────┼─────────┐
   ↓    ↓         ↓
 enc  erb_dec   df_dec
   └────┼─────────┘
        ↓
DeepFilterNet mask + multiframe DF
        ↓
Audio synthesis
        ↓
Enhanced audio
```

The three learned neural-network components are executed by ONNX Runtime.
The surrounding DSP and deterministic DeepFilterNet operations come from the
pinned DeepFilterNet source tree.

## Dependencies

- Python
- NumPy
- ONNX Runtime
- SoundFile
- PyTorch
- DeepFilterNet source + `libdf`

Use the DeepFilterNet commit:

```text
d375b2d8309e0935d165700c91da9de862a99c31
```

Set:

```bash
export DEEPFILTERNET_ROOT=/path/to/DeepFilterNet/DeepFilterNet
```

## Run

From the repository root:

```bash
python inference/enhance_onnx.py     Demo/noisy_0dB.wav     Demo/enhanced_onnx.wav
```

The model defaults to:

```text
models/dfnet3_hindi_onnx/
```

Required files:

```text
enc.onnx
erb_dec.onnx
df_dec.onnx
config.ini
version.txt
```

## Validation evidence

The same ONNX-backed architecture was validated on the controlled Hindi +
DEMAND approximately 0-dB sample:

```text
PyTorch output SNR : 20.408592 dB
ONNX output SNR    : 20.408594 dB
Difference         : ≈ 0.000002 dB
```

These values are from that controlled sample only.

## Deployment status

This is the reference ONNX-backed inference implementation. It is **not yet
a Raspberry Pi 5 benchmark** and makes no real-time performance claim.

The next hardware stage is to measure:

- latency
- real-time factor (RTF)
- CPU utilization
- RAM usage
- continuous streaming stability

A later optimization stage can remove the remaining PyTorch dependency by
porting the deterministic tensor operations to a lighter NumPy/libDF runtime.
