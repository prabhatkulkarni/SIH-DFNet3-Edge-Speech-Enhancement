# SIH-DFNet3 Edge Speech Enhancement

> **Fine-tuned DeepFilterNet3 for multilingual speech enhancement with ONNX-based edge deployment targeting Raspberry Pi 5.**

---

## 🚀 Project Overview

This project develops an AI-based speech enhancement system designed to suppress diverse environmental and mechanical noise while preserving speech intelligibility.

The system uses **DeepFilterNet3**, fine-tuned on multilingual speech and a diverse noise corpus, followed by **ONNX export** for edge deployment.

### Target Application

```text
🎤 Microphone
      ↓
Noisy Speech
      ↓
DeepFilterNet3
      ↓
ONNX Runtime
      ↓
Enhanced Speech
      ↓
🔊 Speaker
