from __future__ import annotations

import numpy as np

from speech2rtmri_artspeech.metrics import compute_contour_metrics


def test_contour_metrics_simple_shift() -> None:
    target = {
        "tongue": np.zeros((3, 50, 2), dtype=np.float32),
    }
    pred = {
        "tongue": np.ones((3, 50, 2), dtype=np.float32),
    }
    metrics = compute_contour_metrics(target, pred, pixel_spacing_mm=1.62)
    assert metrics["rmse_px_mean"] is not None
    assert metrics["rmse_mm_mean"] is not None
    assert "tongue" in metrics["per_articulator"]
