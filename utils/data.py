import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    use_path = False


@dataclass
class SignalStats:
    mean: np.ndarray
    std: np.ndarray


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class EnsureShape:
    def __init__(self, length: int, channels: int = 2):
        self.length = int(length)
        self.channels = int(channels)

    def __call__(self, signal):
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim == 1:
            if array.size % self.channels != 0:
                raise ValueError(
                    f"1D signal of length {array.size} cannot be reshaped into {self.channels} channels."
                )
            array = array.reshape(self.channels, -1)
        elif array.ndim == 2:
            if array.shape[0] == self.channels:
                pass
            elif array.shape[1] == self.channels:
                array = array.T
            else:
                raise ValueError(
                    f"Expected signal shaped [2, L] or [L, 2], got {array.shape}."
                )
        else:
            raise ValueError(f"Unsupported signal rank {array.ndim} for IQ data.")

        cur_len = array.shape[1]
        if cur_len > self.length:
            start = max((cur_len - self.length) // 2, 0)
            array = array[:, start : start + self.length]
        elif cur_len < self.length:
            pad = self.length - cur_len
            left = pad // 2
            right = pad - left
            array = np.pad(array, ((0, 0), (left, right)), mode="constant")

        return np.ascontiguousarray(array, dtype=np.float32)


class EnsureImageShape:
    def __init__(self, channels: int = 3, size: int = 32):
        self.channels = int(channels)
        self.size = int(size)

    def __call__(self, image):
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {array.shape}.")
        if array.shape[0] == self.channels:
            pass
        elif array.shape[-1] == self.channels:
            array = array.transpose(2, 0, 1)
        else:
            raise ValueError(f"Expected [C,H,W] or [H,W,C] with {self.channels} channels, got {array.shape}.")
        if array.shape[1] != self.size or array.shape[2] != self.size:
            raise ValueError(f"Expected image size {self.size}x{self.size}, got {array.shape[1:]}.")
        return np.ascontiguousarray(array, dtype=np.float32)


class RandomTimeShift:
    def __init__(self, max_shift: int):
        self.max_shift = int(max_shift)

    def __call__(self, signal):
        if self.max_shift <= 0:
            return signal
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(signal, shift=shift, axis=1)


class RandomAmplitudeScale:
    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, signal):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return signal * scale


class RandomPhaseRotation:
    def __init__(self, max_degrees: float = 10.0):
        self.max_radians = np.deg2rad(float(max_degrees))

    def __call__(self, signal):
        theta = np.random.uniform(-self.max_radians, self.max_radians)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        i_channel = signal[0] * cos_theta - signal[1] * sin_theta
        q_channel = signal[0] * sin_theta + signal[1] * cos_theta
        return np.stack([i_channel, q_channel], axis=0).astype(np.float32, copy=False)


class AdditiveGaussianNoise:
    def __init__(self, sigma: float = 0.01):
        self.sigma = float(sigma)

    def __call__(self, signal):
        if self.sigma <= 0:
            return signal
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=signal.shape).astype(np.float32)
        return signal + noise


class RandomCropAndPad:
    def __init__(self, crop_ratio: float = 0.95):
        self.crop_ratio = float(crop_ratio)

    def __call__(self, signal):
        if self.crop_ratio >= 1.0:
            return signal
        length = signal.shape[1]
        cropped_len = max(8, int(round(length * self.crop_ratio)))
        if cropped_len >= length:
            return signal
        start = np.random.randint(0, length - cropped_len + 1)
        cropped = signal[:, start : start + cropped_len]
        pad = length - cropped_len
        left = np.random.randint(0, pad + 1)
        right = pad - left
        return np.pad(cropped, ((0, 0), (left, right)), mode="constant").astype(np.float32, copy=False)


class RandomCrop2D:
    def __init__(self, size: int = 32, padding: int = 4):
        self.size = int(size)
        self.padding = int(padding)

    def __call__(self, image):
        padded = np.pad(image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        h, w = padded.shape[1:]
        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)
        return padded[:, top : top + self.size, left : left + self.size]


class RandomHorizontalFlip2D:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image):
        if np.random.rand() < self.p:
            return np.flip(image, axis=2).copy()
        return image


class RandomBrightness2D:
    def __init__(self, max_delta: float = 0.1):
        self.max_delta = float(max_delta)

    def __call__(self, image):
        delta = np.random.uniform(-self.max_delta, self.max_delta)
        return image + delta


class Normalize:
    def __init__(self, mean: Iterable[float], std: Iterable[float], eps: float = 1e-6):
        self.mean = np.asarray(list(mean), dtype=np.float32).reshape(-1, 1)
        self.std = np.maximum(np.asarray(list(std), dtype=np.float32).reshape(-1, 1), eps)

    def __call__(self, signal):
        return (signal - self.mean) / self.std


class NormalizeImage:
    def __init__(self, mean: Iterable[float], std: Iterable[float], eps: float = 1e-6):
        self.mean = np.asarray(list(mean), dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.maximum(np.asarray(list(std), dtype=np.float32).reshape(-1, 1, 1), eps)

    def __call__(self, image):
        return (image - self.mean) / self.std


class ToTensor:
    def __call__(self, signal):
        return torch.from_numpy(np.asarray(signal, dtype=np.float32))


class iADSBIQ(iData):
    def __init__(self, data_root=None, metadata_file=None, iq_len=1024, num_channels=2):
        self.data_root = data_root or "./data"
        self.metadata_file = metadata_file
        self.iq_len = int(iq_len)
        self.num_channels = int(num_channels)
        self.shape_transform = EnsureShape(self.iq_len, self.num_channels)
        self.train_trsf = [
            RandomTimeShift(max_shift=max(1, self.iq_len // 32)),
            RandomAmplitudeScale(0.9, 1.1),
            AdditiveGaussianNoise(0.01),
            RandomPhaseRotation(10.0),
            RandomCropAndPad(0.95),
        ]
        self.test_trsf = []
        self.common_trsf = []
        self.class_order = []

    def download_data(self):
        rows = _load_metadata(self._resolve_metadata_path())
        train_rows = [row for row in rows if row["split"].lower() == "train"]
        test_rows = [row for row in rows if row["split"].lower() == "test"]
        if not train_rows or not test_rows:
            raise ValueError("Metadata must contain both train and test splits.")

        raw_labels = sorted({row["label"] for row in rows})
        label_map = {label: idx for idx, label in enumerate(raw_labels)}
        self.train_data = self._load_signals(train_rows)
        self.test_data = self._load_signals(test_rows)
        self.train_targets = np.asarray([label_map[row["label"]] for row in train_rows], dtype=np.int64)
        self.test_targets = np.asarray([label_map[row["label"]] for row in test_rows], dtype=np.int64)

        stats = _compute_signal_stats(self.train_data)
        self.common_trsf = [Normalize(stats.mean, stats.std), ToTensor()]
        self.class_order = list(range(len(raw_labels)))

    def _resolve_metadata_path(self):
        if not self.metadata_file:
            raise ValueError("ADS-B IQ dataset requires 'metadata_file' in the config.")
        return self.metadata_file if os.path.isabs(self.metadata_file) else os.path.join(self.data_root, self.metadata_file)

    def _load_signals(self, rows):
        signals: List[np.ndarray] = []
        for row in rows:
            signal_path = row["signal_path"]
            if not os.path.isabs(signal_path):
                signal_path = os.path.join(self.data_root, signal_path)
            signal = _read_array_file(signal_path, preferred_key="iq")
            signals.append(self.shape_transform(signal))
        return np.stack(signals, axis=0).astype(np.float32, copy=False)


class iADSBImage(iData):
    def __init__(
        self,
        data_root=None,
        metadata_file=None,
        train_data_file=None,
        train_label_file=None,
        test_data_file=None,
        test_label_file=None,
        image_size=32,
        num_channels=3,
    ):
        self.data_root = data_root or "./data"
        self.metadata_file = metadata_file
        self.train_data_file = train_data_file
        self.train_label_file = train_label_file
        self.test_data_file = test_data_file
        self.test_label_file = test_label_file
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.shape_transform = EnsureImageShape(self.num_channels, self.image_size)
        self.train_trsf = [
            RandomCrop2D(size=self.image_size, padding=4),
            RandomHorizontalFlip2D(0.5),
            RandomBrightness2D(0.1),
        ]
        self.test_trsf = []
        self.common_trsf = []
        self.class_order = []

    def download_data(self):
        if self.metadata_file:
            rows = _load_metadata(self._resolve_path(self.metadata_file))
            train_rows = [row for row in rows if row["split"].lower() == "train"]
            test_rows = [row for row in rows if row["split"].lower() == "test"]
            if not train_rows or not test_rows:
                raise ValueError("Metadata must contain both train and test splits.")
            raw_labels = sorted({row["label"] for row in rows})
            label_map = {label: idx for idx, label in enumerate(raw_labels)}
            self.train_data = self._load_images_from_rows(train_rows)
            self.test_data = self._load_images_from_rows(test_rows)
            self.train_targets = np.asarray([label_map[row["label"]] for row in train_rows], dtype=np.int64)
            self.test_targets = np.asarray([label_map[row["label"]] for row in test_rows], dtype=np.int64)
            self.class_order = list(range(len(raw_labels)))
        else:
            required = [self.train_data_file, self.train_label_file, self.test_data_file, self.test_label_file]
            if any(item is None for item in required):
                raise ValueError(
                    "ADS-B image dataset requires either metadata_file or train/test data and label files."
                )
            self.train_data = self._load_image_batch(self._resolve_path(self.train_data_file))
            self.test_data = self._load_image_batch(self._resolve_path(self.test_data_file))
            self.train_targets = self._load_labels(self._resolve_path(self.train_label_file))
            self.test_targets = self._load_labels(self._resolve_path(self.test_label_file))
            raw_labels = sorted(set(self.train_targets.tolist()) | set(self.test_targets.tolist()))
            label_map = {label: idx for idx, label in enumerate(raw_labels)}
            self.train_targets = np.asarray([label_map[label] for label in self.train_targets], dtype=np.int64)
            self.test_targets = np.asarray([label_map[label] for label in self.test_targets], dtype=np.int64)
            self.class_order = list(range(len(raw_labels)))

        stats = _compute_image_stats(self.train_data)
        self.common_trsf = [NormalizeImage(stats.mean, stats.std), ToTensor()]

    def _resolve_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.data_root, path)

    def _load_images_from_rows(self, rows):
        images: List[np.ndarray] = []
        for row in rows:
            image_path = self._resolve_path(row["signal_path"])
            image = _read_array_file(image_path, preferred_key="image")
            images.append(self.shape_transform(image))
        return np.stack(images, axis=0).astype(np.float32, copy=False)

    def _load_image_batch(self, path):
        array = _read_array_file(path, preferred_key="images")
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError(f"Expected batched image array with 4 dims, got {array.shape}.")
        if array.shape[1] == self.num_channels:
            pass
        elif array.shape[-1] == self.num_channels:
            array = array.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected image batch as [N,C,H,W] or [N,H,W,C], got {array.shape}.")
        if array.shape[2] != self.image_size or array.shape[3] != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {array.shape[2:]}.")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _load_labels(self, path):
        labels = _read_array_file(path, preferred_key="labels")
        labels = np.asarray(labels).reshape(-1)
        return labels


def _load_metadata(metadata_path: str):
    _, ext = os.path.splitext(metadata_path)
    ext = ext.lower()
    if ext == ".csv":
        with open(metadata_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif ext in {".json", ".jsonl"}:
        with open(metadata_path, encoding="utf-8") as handle:
            if ext == ".json":
                rows = json.load(handle)
            else:
                rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported metadata format {ext}. Use .csv, .json, or .jsonl.")

    required = {"signal_path", "label", "split"}
    for row in rows:
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Metadata row missing fields: {sorted(missing)}")
    return rows


def _read_array_file(path: str, preferred_key: str = None):
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".npz":
        payload = np.load(path, allow_pickle=False)
        if preferred_key and preferred_key in payload:
            return payload[preferred_key]
        if len(payload.files) != 1:
            raise ValueError(f"{path} has multiple arrays; expected one or key '{preferred_key}'.")
        return payload[payload.files[0]]
    raise ValueError(f"Unsupported array file format {ext} for {path}.")


def _compute_signal_stats(train_data: np.ndarray):
    flattened = train_data.transpose(1, 0, 2).reshape(train_data.shape[1], -1)
    return SignalStats(mean=flattened.mean(axis=1), std=flattened.std(axis=1))



def _compute_image_stats(train_data: np.ndarray):
    flattened = train_data.transpose(1, 0, 2, 3).reshape(train_data.shape[1], -1)
    return SignalStats(mean=flattened.mean(axis=1), std=flattened.std(axis=1))
