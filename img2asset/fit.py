"""CAD 치수 피팅: 8시점 IoU + 깊이 + 노멀 오차를 최소화한다."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .cameras import RingConfig, ring_cameras
from .losses import LossWeights, ViewLoss, standardize, view_loss
from .model_io import load_model, tessellate
from .render import render_views
from .views import normalize_view

BUILD_FAIL_PENALTY = 10.0
# 렌더 해상도 / 캔버스 해상도. 1.25에서는 얇은 측면 실루엣의 앨리어싱이 손실 바닥을 지배한다(측정값은 README 참고).
DEFAULT_OVERSAMPLE = 1.75


@dataclass
class Target:
    cov: np.ndarray  # 0~1 커버리지
    mask: np.ndarray  # cov >= 0.5
    that: np.ndarray  # 표준화된 상대 깊이 (클수록 카메라에 가까움)


def load_targets(target_dir: str | Path, cfg: RingConfig, resolution: int) -> list[Target]:
    """target_dir/mask_{i}.png 와 depth_{i}.npy (상대 역깊이, 클수록 가까움) 를 읽는다."""
    from PIL import Image

    d = Path(target_dir)
    out = []
    for i in range(cfg.n_views):
        m = np.asarray(Image.open(d / f"mask_{i}.png").convert("L")) > 127
        t = np.load(d / f"depth_{i}.npy").astype(np.float64)
        if t.shape != m.shape:
            t = np.asarray(Image.fromarray(t.astype(np.float32)).resize(m.shape[::-1], Image.BILINEAR))
        cov, nt, _ = normalize_view(m, t, resolution, cfg.frame_frac)
        nm = cov >= 0.5
        out.append(Target(cov, nm, standardize(nt, nm)))
    return out


@dataclass
class EvalResult:
    total: float
    views: list[ViewLoss] = field(default_factory=list)
    error: str | None = None


def evaluate(
    model, params: dict, targets: list[Target], cfg: RingConfig, resolution: int, w: LossWeights,
    oversample: float = DEFAULT_OVERSAMPLE,
) -> EvalResult:
    try:
        part = model.build(params)
        V, T = tessellate(part)
    except Exception as e:  # CAD 커널 실패(필렛 등)는 큰 벌점으로 처리
        return EvalResult(BUILD_FAIL_PENALTY, error=f"{type(e).__name__}: {e}")
    cams = ring_cameras(cfg)
    renders = render_views(V, T, cams, int(resolution * oversample))
    views = []
    for rv, tg in zip(renders, targets):
        cov, h, s = normalize_view(rv.mask, rv.height, resolution, cfg.frame_frac, rv.frame)
        views.append(view_loss(cov, h, rv.pixel_size / s, tg.cov, tg.that, w))
    return EvalResult(float(np.mean([v.total for v in views])), views)


def fit(
    model_path: str | Path,
    target_dir: str | Path,
    out_dir: str | Path,
    cfg: RingConfig | None = None,
    resolution: int = 160,
    maxfev: int = 400,
    weights: LossWeights | None = None,
    init_params: dict | None = None,
    fixed: dict | None = None,
    oversample: float = DEFAULT_OVERSAMPLE,
    log=print,
) -> dict:
    """fixed: 최적화하지 않고 고정할 치수. 이미지에는 절대 스케일이 없으므로
    실제 제품 치수 하나(예: height)를 반드시 고정해야 나머지 치수가 실제 크기로 나온다."""
    cfg = cfg or RingConfig()
    w = weights or LossWeights()
    model = load_model(model_path)
    targets = load_targets(target_dir, cfg, resolution)
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(model.PARAMS)
    if unknown:
        raise ValueError(f"PARAMS에 없는 고정 치수: {sorted(unknown)}")
    if not fixed:
        log("경고: 고정 치수가 없습니다. 결과는 형상 비율만 의미가 있고 절대 크기는 임의입니다.")
    names = [k for k in model.PARAMS if k not in fixed]
    lo = np.array([model.PARAMS[k][1] for k in names], float)
    hi = np.array([model.PARAMS[k][2] for k in names], float)
    x0 = np.array([(init_params or {}).get(k, model.PARAMS[k][0]) for k in names], float)
    span = np.where(hi > lo, hi - lo, 1.0)

    def to_params(z):
        return dict(zip(names, (lo + np.clip(z, 0, 1) * span).tolist())) | fixed

    best = {"loss": np.inf, "z": (x0 - lo) / span}
    n = {"fev": 0}
    t0 = time.time()

    def objective(z):
        n["fev"] += 1
        r = evaluate(model, to_params(z), targets, cfg, resolution, w, oversample)
        if r.total < best["loss"]:
            best.update(loss=r.total, z=np.array(z))
            log(f"[{n['fev']:4d}] loss={r.total:.4f}  ({time.time() - t0:.0f}s)")
        return r.total

    res = minimize(
        objective,
        best["z"].copy(),
        method="Powell",
        bounds=[(0.0, 1.0)] * len(names),
        options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4},
    )
    params = to_params(best["z"])
    final = evaluate(model, params, targets, cfg, resolution, w, oversample)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": str(model_path),
        "params": params,
        "fixed": fixed,
        "loss": final.total,
        "function_evals": n["fev"],
        "optimizer_message": str(res.message),
        "views": [
            {"view": i, "azimuth_deg": c.azimuth_deg, "iou": v.iou, "depth_huber": v.depth, "normal_err_deg": v.normal_deg}
            for i, (c, v) in enumerate(zip(ring_cameras(cfg), final.views))
        ],
    }
    (out / "params.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    from .report import write_report

    write_report(final, targets, cfg, out, params)
    export_part(model, params, out)
    return summary


def export_part(model, params: dict, out: Path) -> None:
    import build123d as bd

    part = model.build(params)
    bd.export_step(part, str(out / "model.step"))
    bd.export_stl(part, str(out / "model.stl"))
