"""합성 타깃 생성: 정답 치수로 렌더한 깊이를 '추정 깊이'처럼 바꿔 저장한다.

실제 이미지 없이 피팅이 수렴하는지 검증하는 용도다.
추정 모델의 특성을 흉내 내려고 임의의 scale/offset과 노이즈를 넣는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cameras import RingConfig, ring_cameras
from .model_io import load_model, tessellate
from .render import render_views


def make_synthetic_targets(
    model_path: str | Path,
    params: dict,
    out_dir: str | Path,
    cfg: RingConfig | None = None,
    resolution: int = 256,
    scale: float = 3.7,
    offset: float = 12.0,
    noise: float = 0.0,
    seed: int = 0,
) -> None:
    from PIL import Image

    cfg = cfg or RingConfig()
    model = load_model(model_path)
    full = {k: v[0] for k, v in model.PARAMS.items()} | params
    V, T = tessellate(model.build(full))
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, rv in enumerate(render_views(V, T, ring_cameras(cfg), resolution)):
        h = rv.height
        t = np.where(rv.mask, scale * h + offset + noise * rng.standard_normal(h.shape), 0.0)
        np.save(out / f"depth_{i}.npy", t.astype(np.float32))
        Image.fromarray((rv.mask * 255).astype(np.uint8)).save(out / f"mask_{i}.png")
    (out / "ground_truth.json").write_text(json.dumps(full, indent=2))
