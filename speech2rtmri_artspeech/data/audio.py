from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

from ..common import ensure_dir, read_jsonl, write_json


@dataclass(frozen=True)
class AudioEncoderSpec:
    encoder: str
    model_name: str
    sample_rate: int
    embedding_dim: int
    device: str


def _sha1_for_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


class SessionAudioEmbedder:
    MODEL_MAP = {
        "wavlm": "microsoft/wavlm-base",
        "hubert": "facebook/hubert-base-ls960",
        "wav2vec2": "facebook/wav2vec2-base",
    }

    def __init__(self, config: dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.audio_cfg = config["audio"]
        self.encoder = self.audio_cfg["encoder"]
        self.device = self.audio_cfg.get("device", "cpu")
        self.sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        self.embedding_dim = int(self.audio_cfg["embedding_dim"])
        self._feature_extractor = None
        self._model = None

    def encode_session(self, wav_path: str | Path) -> dict[str, Any]:
        wav_path = Path(wav_path)
        audio = _load_audio(wav_path, self.sample_rate)

        if self.encoder == "mfcc":
            return self._encode_mfcc(audio, wav_path)
        return self._encode_transformer(audio, wav_path)

    def _encode_mfcc(self, audio: np.ndarray, wav_path: Path) -> dict[str, Any]:
        hop_length = int(self.audio_cfg.get("hop_length", 320))
        win_length = int(self.audio_cfg.get("win_length", 400))
        n_fft = int(self.audio_cfg.get("n_fft", 400))
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.embedding_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        ).T.astype(np.float32)
        stride_sec = hop_length / self.sample_rate
        times = np.arange(len(mfcc), dtype=np.float32) * stride_sec + stride_sec / 2.0
        return {
            "embeddings": torch.from_numpy(mfcc),
            "times_sec": torch.from_numpy(times),
            "metadata": {
                "encoder": self.encoder,
                "model_name": "mfcc",
                "sample_rate": self.sample_rate,
                "stride_sec": stride_sec,
                "wav_sha1": _sha1_for_file(wav_path),
            },
        }

    def _encode_transformer(self, audio: np.ndarray, wav_path: Path) -> dict[str, Any]:
        if self._feature_extractor is None or self._model is None:
            self._lazy_load_model()

        inputs = self._feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            outputs = self._model(input_values)
        embeddings = outputs.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32)

        conv_stride = getattr(self._model.config, "conv_stride", None)
        if conv_stride is None:
            raise ValueError(f"Model {self.encoder} does not expose conv_stride for timestamp alignment")
        stride_sec = float(np.prod(conv_stride)) / float(self.sample_rate)
        times = torch.arange(embeddings.shape[0], dtype=torch.float32) * stride_sec + stride_sec / 2.0

        return {
            "embeddings": embeddings,
            "times_sec": times,
            "metadata": {
                "encoder": self.encoder,
                "model_name": self.MODEL_MAP[self.encoder],
                "sample_rate": self.sample_rate,
                "stride_sec": stride_sec,
                "wav_sha1": _sha1_for_file(wav_path),
            },
        }

    def _lazy_load_model(self) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for wavlm / hubert / wav2vec2 audio embeddings"
            ) from exc

        model_name = self.audio_cfg.get("model_name") or self.MODEL_MAP[self.encoder]
        self.logger.info("Loading audio encoder %s (%s)", self.encoder, model_name)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device)
        self._model.eval()


def extract_audio_embeddings(config: dict[str, Any], logger, force: bool = False) -> list[str]:
    output_root = Path(config["runtime"]["output_dir"])
    manifest_dir = output_root / "manifests"
    cache_root = output_root / "audio_cache"
    encoder = config["audio"]["encoder"]
    embedder = SessionAudioEmbedder(config, logger)

    unique_sessions: dict[tuple[str, str], dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        manifest_path = manifest_dir / f"{split_name}.jsonl"
        if not manifest_path.exists():
            continue
        for row in read_jsonl(manifest_path):
            unique_sessions[(row["speaker"], row["session"])] = row

    saved_paths: list[str] = []
    for (speaker, session), row in sorted(unique_sessions.items()):
        cache_path = cache_root / encoder / speaker / f"{session}.pt"
        meta_path = cache_root / encoder / speaker / f"{session}.json"
        if cache_path.exists() and meta_path.exists() and not force:
            logger.info("Skipping cached audio embeddings for %s/%s", speaker, session)
            saved_paths.append(str(cache_path))
            continue

        logger.info("Encoding audio session %s/%s with %s", speaker, session, encoder)
        payload = embedder.encode_session(row["wav_path"])
        ensure_dir(cache_path.parent)
        torch.save(payload, cache_path)
        write_json(meta_path, payload["metadata"])
        saved_paths.append(str(cache_path))

    return saved_paths
