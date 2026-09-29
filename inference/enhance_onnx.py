#!/usr/bin/env python3
"""ONNX-backed DeepFilterNet3 inference for the SIH model."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort


def add_df_root(root):
    if root:
        root = os.path.abspath(root)
        if root not in sys.path:
            sys.path.insert(0, root)


class OnnxEncoder(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.outputs = [x.name for x in self.session.get_outputs()]

    @torch.no_grad()
    def forward(self, feat_erb, feat_spec):
        values = self.session.run(
            self.outputs,
            {
                "feat_erb": feat_erb.detach().cpu().numpy().astype(np.float32),
                "feat_spec": feat_spec.detach().cpu().numpy().astype(np.float32),
            },
        )
        out = dict(zip(self.outputs, values))
        names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
        return tuple(torch.from_numpy(out[n]) for n in names)


class OnnxErbDecoder(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.outputs = [x.name for x in self.session.get_outputs()]

    @torch.no_grad()
    def forward(self, emb, e3, e2, e1, e0):
        tensors = {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0}
        values = self.session.run(
            self.outputs,
            {
                n: tensors[n].detach().cpu().numpy().astype(np.float32)
                for n in [x.name for x in self.session.get_inputs()]
            },
        )
        return torch.from_numpy(values[self.outputs.index("m")])


class OnnxDfDecoder(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.outputs = [x.name for x in self.session.get_outputs()]

    @torch.no_grad()
    def forward(self, emb, c0):
        tensors = {"emb": emb, "c0": c0}
        values = self.session.run(
            self.outputs,
            {
                n: tensors[n].detach().cpu().numpy().astype(np.float32)
                for n in [x.name for x in self.session.get_inputs()]
            },
        )
        return torch.from_numpy(values[self.outputs.index("coefs")])


def build_model(model_dir, df_root):
    add_df_root(df_root)

    from df.config import config
    from df.deepfilternet3 import init_model
    from df.model import ModelParams
    from libdf import DF

    config.load(
        str(model_dir / "config.ini"),
        config_must_exist=True,
        allow_defaults=False,
        allow_reload=True,
    )

    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )

    # Only deterministic DeepFilterNet operations remain in PyTorch.
    # The learned encoder/decoders are replaced by the exported ONNX models.
    model = init_model(df_state=df_state, run_df=True, train_mask=True).cpu().eval()
    model.enc = OnnxEncoder(model_dir / "enc.onnx")
    model.erb_dec = OnnxErbDecoder(model_dir / "erb_dec.onnx")
    model.df_dec = OnnxDfDecoder(model_dir / "df_dec.onnx")

    return model, df_state


def enhance(model, df_state, audio):
    from df.enhance import df_features
    from df.model import ModelParams
    from df.utils import as_complex

    x = torch.from_numpy(audio.astype(np.float32)).reshape(1, -1)
    original_len = x.shape[-1]
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()

    x = torch.nn.functional.pad(x, (0, n_fft))
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))

    spec, erb_feat, spec_feat = df_features(
        x, df_state, nb_df, device="cpu"
    )

    with torch.no_grad():
        enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()

    enhanced = as_complex(enhanced.squeeze(1))
    out = torch.as_tensor(df_state.synthesis(enhanced.numpy()))

    # Same delay compensation as the validated DeepFilterNet enhance path.
    out = out[:, (n_fft - hop):(original_len + n_fft - hop)]
    return out.squeeze(0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models" / "dfnet3_hindi_onnx",
    )
    ap.add_argument(
        "--df-root",
        default=os.environ.get("DEEPFILTERNET_ROOT"),
        help="Path to DeepFilterNet/DeepFilterNet at the pinned commit.",
    )
    args = ap.parse_args()

    required = ["enc.onnx", "erb_dec.onnx", "df_dec.onnx",
                "config.ini", "version.txt"]
    missing = [x for x in required if not (args.model_dir / x).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {args.model_dir}: {missing}"
        )

    audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    model, df_state = build_model(args.model_dir, args.df_root)

    expected_sr = df_state.sr()
    if sr != expected_sr:
        raise ValueError(
            f"Input is {sr} Hz; this model expects {expected_sr} Hz."
        )

    print("ONNX Runtime:", ort.__version__)
    print("Provider: CPUExecutionProvider")
    print("Sample rate:", expected_sr)
    print("Input:", args.input)

    enhanced = enhance(model, df_state, audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced, expected_sr, subtype="PCM_16")

    print("Output:", args.output)
    print("Output samples:", len(enhanced))
    print("Output peak:", float(np.max(np.abs(enhanced))))


if __name__ == "__main__":
    main()
