from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from contextlib import contextmanager


class ImagenVideoDiffusionWrapper:
    def __init__(self, config: dict[str, Any], logger, device: torch.device):
        self.config = config
        self.logger = logger
        self.device = device
        self.imagen = None
        self.trainer = None
        self._has_explicit_valid_dataset = False
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            from imagen_pytorch import ElucidatedImagen, ImagenTrainer, Unet3D
        except ImportError as exc:
            raise ImportError(
                "imagen-pytorch is required for diffusion training and sampling. "
                "Install the optional 'diffusion' dependencies in the repo-local env."
            ) from exc

        model_cfg = self.config["model"]
        frame_size = int(self.config["data"]["frame_size"])
        audio_embed_dim = int(self.config["audio"]["embedding_dim"])
        dim_mults = tuple(model_cfg["dim_mults"])
        channels = int(model_cfg.get("channels", 1))

        unet_kwargs = {
            "dim": int(model_cfg["dim"]),
            "dim_mults": dim_mults,
            "channels": channels,
        }
        for optional_key in (
            "memory_efficient",
            "attn_heads",
            "attn_dim_head",
            "ff_mult",
            "use_linear_attn",
            "use_linear_cross_attn",
            "layer_attns",
            "layer_cross_attns",
        ):
            if optional_key in model_cfg:
                unet_kwargs[optional_key] = model_cfg[optional_key]

        unet = Unet3D(**unet_kwargs)
        self.imagen = ElucidatedImagen(
            text_embed_dim=audio_embed_dim,
            channels=channels,
            unets=(unet,),
            image_sizes=frame_size,
            temporal_downsample_factor=1,
            num_sample_steps=int(model_cfg["num_sample_steps"]),
            cond_drop_prob=float(model_cfg["cond_drop_prob"]),
            sigma_min=float(model_cfg["sigma_min"]),
            sigma_max=float(model_cfg["sigma_max"]),
            sigma_data=float(model_cfg["sigma_data"]),
            rho=float(model_cfg["rho"]),
            P_mean=float(model_cfg["P_mean"]),
            P_std=float(model_cfg["P_std"]),
            S_churn=float(model_cfg["S_churn"]),
            S_tmin=float(model_cfg["S_tmin"]),
            S_tmax=float(model_cfg["S_tmax"]),
            S_noise=float(model_cfg["S_noise"]),
        ).to(self.device)

        requested_amp = bool(self.config["train"].get("use_amp", False) and self.device.type == "cuda")
        if requested_amp:
            self.logger.warning(
                "Mixed precision was requested but is temporarily disabled because "
                "the current imagen-pytorch ElucidatedImagen path asserts float32 images "
                "after the trainer casts tensors to float16."
            )

        self.trainer = ImagenTrainer(
            self.imagen,
            lr=float(self.config["train"]["learning_rate"]),
            split_valid_from_train=False,
            dl_tuple_output_keywords_names=("images", "text_embeds", "cond_video_frames"),
            fp16=False,
        ).to(self.device)

    def register_datasets(self, train_dataset, train_batch_size: int, valid_dataset=None, valid_batch_size: int | None = None) -> None:
        self.trainer.add_train_dataset(train_dataset, batch_size=train_batch_size)
        if valid_dataset is not None and hasattr(self.trainer, "add_valid_dataset"):
            self.trainer.add_valid_dataset(valid_dataset, batch_size=valid_batch_size or train_batch_size)
            self._has_explicit_valid_dataset = True

    def train_step(self, *, unet_number: int = 1, max_batch_size: int = 1, ignore_time: bool = False) -> float:
        loss = self.trainer.train_step(
            unet_number=unet_number,
            max_batch_size=max_batch_size,
            ignore_time=ignore_time,
        )
        return float(loss)

    def valid_step(self, *, unet_number: int = 1, max_batch_size: int = 1, ignore_time: bool = False) -> float:
        loss = self.trainer.valid_step(
            unet_number=unet_number,
            max_batch_size=max_batch_size,
            ignore_time=ignore_time,
        )
        return float(loss)

    def sample(
        self,
        *,
        text_embeds: torch.Tensor,
        cond_video_frames: torch.Tensor,
        video_frames: int,
        stop_at_unet_number: int = 1,
        cond_scale: float = 1.0,
        num_sample_steps: int | None = None,
    ) -> torch.Tensor:
        text_embeds = text_embeds.to(self.device)
        cond_video_frames = cond_video_frames.to(self.device)
        with self._override_sample_steps(stop_at_unet_number, num_sample_steps):
            return self.trainer.sample(
                text_embeds=text_embeds,
                cond_video_frames=cond_video_frames,
                video_frames=video_frames,
                stop_at_unet_number=stop_at_unet_number,
                batch_size=text_embeds.shape[0],
                cond_scale=cond_scale,
            )

    @contextmanager
    def _override_sample_steps(self, unet_number: int, num_sample_steps: int | None):
        if num_sample_steps is None:
            yield
            return

        unet_index = unet_number - 1
        original_hparams = self.imagen.hparams[unet_index]
        self.imagen.hparams[unet_index] = original_hparams._replace(num_sample_steps=int(num_sample_steps))
        try:
            yield
        finally:
            self.imagen.hparams[unet_index] = original_hparams

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save(str(checkpoint_path))

    def load(self, checkpoint_path: str | Path) -> None:
        self.trainer.load(str(checkpoint_path))

    def num_steps_taken(self, unet_number: int = 1) -> int:
        return int(self.trainer.num_steps_taken(unet_number=unet_number))
