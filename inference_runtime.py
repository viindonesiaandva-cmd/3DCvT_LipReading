# -*- coding: utf-8 -*-
"""Shared runtime utilities for single, batch, and service inference."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils import DATASET_REGISTRY, LipReading3DCvT, load_weights


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionRequest:
    """Describes one inference input."""

    input_path: str
    input_type: str = 'auto'


@dataclass(frozen=True)
class TopKPrediction:
    """Represents one ranked prediction item."""

    rank: int
    word: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output for one sample."""

    input_path: str
    input_type: str
    top_prediction: str
    top_confidence: float
    predictions: List[TopKPrediction]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class PredictionFailure:
    """Captures an input that could not be processed."""

    input_path: str
    input_type: str
    error: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class _PreparedSample:
    """Holds tensors for one already-processed request."""

    input_path: str
    input_type: str
    video_tensor: torch.Tensor
    boundary_mask: torch.Tensor


class InferenceProcessor:
    """Converts raw videos or preprocessed PKL files into model inputs."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        crop_size: int = 88,
        max_len: int = 29,
        mean: float = 0.421,
        std: float = 0.165,
    ):
        self.target_size = target_size
        self.crop_size = crop_size
        self.max_len = max_len
        self.mean = mean
        self.std = std

    def process_video(self, video_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize a raw video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        if not cap.isOpened():
            raise IOError(f'Cannot open video: {video_path}')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError('Video is empty')

        return self._pack(np.asarray(frames, dtype=np.uint8), boundary=None)

    def process_pkl(self, pkl_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize a preprocessed PKL sample."""
        pkl_path = Path(pkl_path)
        with pkl_path.open('rb') as handle:
            data = pickle.load(handle)

        if 'video' not in data:
            raise KeyError(f"Missing 'video' key in {pkl_path}")

        frames = np.asarray(data['video'])
        if frames.ndim == 4:
            frames = self._convert_to_grayscale(frames, pkl_path)
        if frames.ndim != 3:
            raise ValueError(
                f'Expected video frames with shape (T, H, W), got {frames.shape}'
            )

        return self._pack(frames, boundary=data.get('boundary'))

    def _convert_to_grayscale(self, frames: np.ndarray, pkl_path: Path) -> np.ndarray:
        """Convert channel-last PKL frames into grayscale frames."""
        converted = []
        for frame in frames:
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[-1] == 1:
                frame = frame.squeeze(-1)
            else:
                raise ValueError(
                    f'Unsupported frame shape in {pkl_path}: {frame.shape}'
                )
            converted.append(frame)
        return np.stack(converted, axis=0)

    def _pack(
        self,
        frames: np.ndarray,
        boundary: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply center crop, normalization, and temporal padding."""
        frames = self._center_crop(frames)
        frames = self._normalize(frames)
        frames = self._pad_or_truncate(frames)

        video_tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)
        boundary_mask = self._build_boundary_mask(boundary, video_tensor.shape[2])
        return video_tensor, boundary_mask

    def _center_crop(self, frames: np.ndarray) -> np.ndarray:
        """Apply the validation-time center crop used by the dataset."""
        _, height, width = frames.shape
        target_h = self.crop_size
        target_w = self.crop_size
        if height < target_h or width < target_w:
            raise ValueError(
                f'Frame size ({height}x{width}) is smaller than crop size '
                f'({target_h}x{target_w})'
            )

        x1 = int(round((width - target_w) / 2.0))
        y1 = int(round((height - target_h) / 2.0))
        return frames[:, y1:y1 + target_h, x1:x1 + target_w]

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        """Apply the dataset mean/std normalization."""
        frames = frames.astype(np.float32) / 255.0
        return (frames - self.mean) / self.std

    def _pad_or_truncate(self, frames: np.ndarray) -> np.ndarray:
        """Force a fixed temporal length for model input."""
        time_len = frames.shape[0]
        if time_len > self.max_len:
            return frames[:self.max_len]
        if time_len < self.max_len:
            pad_shape = (self.max_len - time_len, frames.shape[1], frames.shape[2])
            padding = np.zeros(pad_shape, dtype=np.float32)
            return np.concatenate([frames, padding], axis=0)
        return frames

    def _build_boundary_mask(
        self,
        boundary: Optional[Sequence[int]],
        seq_len: int,
    ) -> torch.Tensor:
        """Build a binary boundary mask aligned with the time axis."""
        mask = torch.ones(seq_len, 1, dtype=torch.float32)
        if boundary is None:
            return mask.unsqueeze(0)

        try:
            start = max(0, min(int(boundary[0]), seq_len - 1))
            end = max(start, min(int(boundary[1]), seq_len - 1))
        except (TypeError, ValueError, IndexError):
            return mask.unsqueeze(0)

        mask.zero_()
        mask[start:end + 1] = 1.0
        return mask.unsqueeze(0)


def resolve_dataset_config(
    dataset: str,
    data_root: Optional[str],
    num_classes: Optional[int],
) -> Tuple[str, Path, int]:
    """Resolve dataset defaults while allowing explicit overrides."""
    _, default_root, default_classes = DATASET_REGISTRY[dataset]
    resolved_root = Path(data_root) if data_root is not None else Path(default_root)
    resolved_classes = num_classes if num_classes is not None else default_classes
    return dataset, resolved_root, resolved_classes


def create_device(gpu: str) -> torch.device:
    """Create the runtime device and enable cuDNN autotuning when possible."""
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    return device


def load_vocab(data_root: str | Path) -> List[str]:
    """Build the class list in the same order as the dataset loader."""
    data_root = Path(data_root)
    vocab_path = data_root / 'vocab.json'
    if vocab_path.exists():
        with vocab_path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    return sorted(directory.name for directory in data_root.iterdir() if directory.is_dir())


def load_model(checkpoint_path: str | Path, num_classes: int, device: torch.device) -> LipReading3DCvT:
    """Instantiate the model and load weights onto the target device."""
    model = LipReading3DCvT(num_classes=num_classes)
    load_weights(checkpoint_path, model, torch.device('cpu'))
    model.to(device)
    model.eval()
    return model


class InferenceSession:
    """Owns one model instance and serves repeated inference requests."""

    def __init__(
        self,
        dataset: str,
        checkpoint_path: str | Path,
        data_root: Optional[str] = None,
        num_classes: Optional[int] = None,
        gpu: str = '0',
    ):
        self.dataset, self.data_root, self.num_classes = resolve_dataset_config(
            dataset=dataset,
            data_root=data_root,
            num_classes=num_classes,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.device = create_device(gpu)
        self.processor = InferenceProcessor()

        logger.info('Loading vocabulary...')
        self.vocab = load_vocab(self.data_root)
        if len(self.vocab) != self.num_classes:
            logger.warning(
                'Found %s classes in %s, but model expects %s.',
                len(self.vocab),
                self.data_root,
                self.num_classes,
            )

        logger.info('Loading model...')
        self.model = load_model(self.checkpoint_path, self.num_classes, self.device)

    def predict_video(self, video_path: str | Path, top_k: int = 5) -> PredictionResult:
        """Run inference for one raw video."""
        request = PredictionRequest(input_path=str(video_path), input_type='video')
        return self.predict_request(request, top_k=top_k)

    def predict_pkl(self, pkl_path: str | Path, top_k: int = 5) -> PredictionResult:
        """Run inference for one preprocessed PKL sample."""
        request = PredictionRequest(input_path=str(pkl_path), input_type='pkl')
        return self.predict_request(request, top_k=top_k)

    def predict_request(
        self,
        request: PredictionRequest,
        top_k: int = 5,
    ) -> PredictionResult:
        """Run inference for one request."""
        prepared = self.prepare_request(request)
        return self.predict_prepared_batch([prepared], top_k=top_k)[0]

    def predict_batch(
        self,
        requests: Sequence[PredictionRequest],
        top_k: int = 5,
        batch_size: int = 8,
    ) -> Tuple[List[PredictionResult], List[PredictionFailure]]:
        """Run inference for many requests while loading the model only once."""
        if batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')

        prepared_samples: List[_PreparedSample] = []
        failures: List[PredictionFailure] = []
        for request in requests:
            try:
                prepared_samples.append(self.prepare_request(request))
            except Exception as exc:  # noqa: BLE001 - batch mode should continue on bad inputs.
                failures.append(
                    PredictionFailure(
                        input_path=request.input_path,
                        input_type=self.resolve_input_type(request),
                        error=str(exc),
                    )
                )

        results: List[PredictionResult] = []
        for start in range(0, len(prepared_samples), batch_size):
            chunk = prepared_samples[start:start + batch_size]
            results.extend(self.predict_prepared_batch(chunk, top_k=top_k))
        return results, failures

    def prepare_request(self, request: PredictionRequest) -> _PreparedSample:
        """Load and preprocess one request into tensors."""
        input_type = self.resolve_input_type(request)
        input_path = Path(request.input_path)
        if input_type == 'video':
            video_tensor, boundary_mask = self.processor.process_video(input_path)
        else:
            video_tensor, boundary_mask = self.processor.process_pkl(input_path)
        return _PreparedSample(
            input_path=str(input_path),
            input_type=input_type,
            video_tensor=video_tensor,
            boundary_mask=boundary_mask,
        )

    def resolve_input_type(self, request: PredictionRequest) -> str:
        """Infer the input type when the caller leaves it on auto."""
        input_type = request.input_type.lower()
        if input_type not in {'auto', 'video', 'pkl'}:
            raise ValueError(f'Unsupported input_type: {request.input_type}')
        if input_type != 'auto':
            return input_type

        suffix = Path(request.input_path).suffix.lower()
        if suffix == '.pkl':
            return 'pkl'
        if suffix in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
            return 'video'
        raise ValueError(
            f'Cannot infer input type from path: {request.input_path}. '
            'Please set input_type to video or pkl.'
        )

    def predict_prepared_batch(
        self,
        samples: Sequence[_PreparedSample],
        top_k: int = 5,
    ) -> List[PredictionResult]:
        """Run model forward on already-prepared tensors."""
        if not samples:
            return []

        effective_top_k = min(top_k, self.num_classes, len(self.vocab))
        videos = torch.cat([sample.video_tensor for sample in samples], dim=0)
        boundaries = torch.cat([sample.boundary_mask for sample in samples], dim=0)
        videos = videos.to(self.device, non_blocking=self.device.type == 'cuda')
        boundaries = boundaries.to(self.device, non_blocking=self.device.type == 'cuda')

        with torch.inference_mode():
            logits = self.model(videos, boundaries)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, effective_top_k, dim=1)

        results = []
        for sample, sample_probs, sample_indices in zip(samples, top_probs, top_indices):
            predictions = []
            for rank, (probability, index) in enumerate(
                zip(sample_probs.tolist(), sample_indices.tolist()),
                start=1,
            ):
                predictions.append(
                    TopKPrediction(
                        rank=rank,
                        word=self.vocab[index],
                        confidence=probability * 100.0,
                    )
                )
            results.append(
                PredictionResult(
                    input_path=sample.input_path,
                    input_type=sample.input_type,
                    top_prediction=predictions[0].word,
                    top_confidence=predictions[0].confidence,
                    predictions=predictions,
                )
            )
        return results


def format_prediction_report(result: PredictionResult) -> str:
    """Format one prediction result for console output."""
    lines = [
        '',
        '=' * 30,
        ' PREDICTION RESULTS',
        '=' * 30,
        f'Input: {result.input_path}',
        f"Top Prediction: '{result.top_prediction}' ({result.top_confidence:.2f}%)",
        '-' * 30,
        f"{'Rank':<5} | {'Word':<15} | {'Confidence':<10}",
        '-' * 35,
    ]
    for item in result.predictions:
        lines.append(
            f'{item.rank:<5} | {item.word:<15} | {item.confidence:.2f}%'
        )
    lines.append('=' * 30)
    return '\n'.join(lines)
