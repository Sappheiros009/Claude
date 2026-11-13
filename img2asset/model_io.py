"""CAD 모델 파일 로딩과 메시화.

모델 파일 규약 (Claude Code가 턴어라운드를 보고 작성):

    PARAMS = {"이름": (초깃값, 최솟값, 최댓값), ...}   # 단위 mm
    def build(p: dict) -> build123d.Part: ...

구조(어떤 형상을 더하고 빼는지)는 코드가 정하고, 치수는 PARAMS로 노출해서 피팅이 조정한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def load_model(path: str | Path) -> ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"cadmodel_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PARAMS") or not hasattr(mod, "build"):
        raise ValueError(f"{path}: PARAMS 와 build(p) 가 필요합니다.")
    for k, (init, lo, hi) in mod.PARAMS.items():
        if not lo <= init <= hi:
            raise ValueError(f"{path}: {k} 초깃값 {init}이 범위 [{lo}, {hi}] 밖입니다.")
    return mod


def tessellate(part, rel_tol: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    bb = part.bounding_box()
    diag = float(np.linalg.norm([bb.size.X, bb.size.Y, bb.size.Z]))
    verts, tris = part.tessellate(max(1e-4, rel_tol * diag), 0.2)
    V = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=np.float64)
    T = np.array(tris, dtype=np.int64)
    if len(T) == 0:
        raise ValueError("메시화 결과가 비어 있습니다.")
    return V, T
