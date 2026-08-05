"""Mel spectrogram front-end compatible with PANNs / torchlibrosa (no librosa)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _periodic_hann(length: int) -> np.ndarray:
    """Periodic Hann window (librosa get_window(..., fftbins=True))."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def _pad_center(data: np.ndarray, size: int) -> np.ndarray:
    n = int(data.shape[0])
    if size < n:
        raise ValueError(f"target size {size} smaller than data length {n}")
    left = (size - n) // 2
    right = size - n - left
    return np.pad(data, (left, right), mode="constant")


def _hz_to_mel(frequencies: np.ndarray, htk: bool = False) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    # Slaney
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels = np.asarray(mels)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    # Slaney
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs = np.asarray(freqs)
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Slaney-normalized mel filter bank matching librosa.filters.mel defaults."""
    if fmax is None:
        fmax = float(sr) / 2.0

    n_fft = int(n_fft)
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    fftfreqs = np.linspace(0, float(sr) / 2.0, n_fft // 2 + 1, endpoint=True)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style area normalization
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


class STFT(nn.Module):
    """Conv1d STFT with torchlibrosa parameter names (*.stft.conv_*)."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        freeze_parameters: bool = True,
    ):
        super().__init__()
        if window != "hann":
            raise ValueError("only hann window is supported in the vendored front-end")
        if pad_mode not in ("constant", "reflect"):
            raise ValueError(f"unsupported pad_mode: {pad_mode}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode

        fft_window = _pad_center(_periodic_hann(win_length), n_fft)
        # Match torchlibrosa DFTBase.dft_matrix + STFT weight layout.
        x, y = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
        dft = np.exp(-2j * np.pi * x * y / n_fft)
        out_channels = n_fft // 2 + 1
        windowed = dft[:, :out_channels] * fft_window[:, None]

        self.conv_real = nn.Conv1d(
            1, out_channels, kernel_size=n_fft, stride=hop_length, bias=False
        )
        self.conv_imag = nn.Conv1d(
            1, out_channels, kernel_size=n_fft, stride=hop_length, bias=False
        )
        self.conv_real.weight.data = torch.tensor(
            np.real(windowed).T, dtype=torch.float32
        ).unsqueeze(1)
        self.conv_imag.weight.data = torch.tensor(
            np.imag(windowed).T, dtype=torch.float32
        ).unsqueeze(1)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input: (batch, samples) -> real/imag (batch, 1, time, freq)
        x = input[:, None, :]
        if self.center:
            x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        return real, imag


class Spectrogram(nn.Module):
    """STFT power spectrogram via nested STFT (torchlibrosa-compatible keys)."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        freeze_parameters: bool = True,
    ):
        super().__init__()
        self.power = power
        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(input)
        spectrogram = real**2 + imag**2
        if self.power != 2.0:
            spectrogram = spectrogram ** (self.power / 2.0)
        return spectrogram


class LogmelFilterBank(nn.Module):
    """Log-mel projection matching torchlibrosa / librosa defaults."""

    def __init__(
        self,
        sr: int = 32000,
        n_fft: int = 1024,
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: float = 14000.0,
        is_log: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float | None = None,
        freeze_parameters: bool = True,
    ):
        super().__init__()
        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        mel_w = mel_filter_bank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.melW = nn.Parameter(torch.tensor(mel_w, dtype=torch.float32))
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(input, self.melW)
        if not self.is_log:
            return mel
        return self._power_to_db(mel)

    def _power_to_db(self, input: torch.Tensor) -> torch.Tensor:
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin))
        log_spec = log_spec - 10.0 * float(np.log10(max(self.amin, self.ref)))
        if self.top_db is not None:
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db)
        return log_spec
