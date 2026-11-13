"""M3: 클레이 턴어라운드 이미지에서 CPU로 상대 깊이를 추정한다.

입력은 반드시 무광 클레이 버전 턴어라운드여야 한다(prompts.clay_turnaround_prompt 참고).
광택 이미지의 반사·하이라이트는 깊이 모델이 형상으로 오인한다.

Depth Anything V2 Small을 기본으로 쓴다. 출력은 상대 역깊이(클수록 가까움)이고 스케일과 오프셋이
정해지지 않으므로, 피팅 단계에서 affine 정렬 후 비교한다.
필요 패키지: transformers, torch (CPU 빌드)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate_depth(view_dir: str | Path, out_dir: str | Path, n_views: int = 8, model_id: str = DEFAULT_MODEL) -> None:
    import torch
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model_id, device="cpu")
    src, out = Path(view_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_views):
        img = Image.open(src / f"view_{i}.png").convert("RGB")
        pred = pipe(img)["predicted_depth"]
        pred = pred.reshape(1, 1, *pred.shape[-2:]).float()
        pred = torch.nn.functional.interpolate(pred, size=img.size[::-1], mode="bicubic", align_corners=False)
        np.save(out / f"depth_{i}.npy", pred[0, 0].numpy().astype(np.float32))
        if (src / f"mask_{i}.png").resolve() != (out / f"mask_{i}.png").resolve():
            shutil.copy(src / f"mask_{i}.png", out / f"mask_{i}.png")
        print(f"view {i}: depth 저장")
