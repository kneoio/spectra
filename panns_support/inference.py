"""Sound event detection wrapper (panns-inference compatible API)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .labels import classes_num, labels
from .models import Cnn14_DecisionLevelMax
from .pytorch_utils import move_data_to_device

CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
)
DEFAULT_CHECKPOINT = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size >= 3e8:
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PANNs checkpoint to {checkpoint_path} ...")
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, tmp_path)
        tmp_path.replace(checkpoint_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 3e8:
        raise RuntimeError(
            f"Failed to download PANNs checkpoint to {checkpoint_path}"
        )
    return checkpoint_path


class SoundEventDetection:
    """Framewise AudioSet tagging, API-compatible with panns_inference."""

    def __init__(
        self,
        model=None,
        checkpoint_path: str | os.PathLike | None = None,
        device: str = "cuda",
        interpolate_mode: str = "nearest",
    ):
        path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        path = _ensure_checkpoint(path)
        print(f"Checkpoint path: {path}")

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            if device == "cuda":
                print("Using CPU.")

        self.labels = labels
        self.classes_num = classes_num

        if model is None:
            self.model = Cnn14_DecisionLevelMax(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=self.classes_num,
                interpolate_mode=interpolate_mode,
            )
        else:
            self.model = model

        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:  # torch < 2.0
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        if self.device == "cuda":
            self.model.to(self.device)
            print(f"GPU number: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model)
        else:
            print("Using CPU.")

    def inference(self, audio: np.ndarray) -> np.ndarray:
        audio_t = move_data_to_device(audio, self.device)
        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(input=audio_t, mixup_lambda=None)
        return output_dict["framewise_output"].data.cpu().numpy()
