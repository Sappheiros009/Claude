"""Phase 1 회귀 테스트 (합성 타깃, CPU).

실행: pytest -q tests/test_phase1.py   (4코어 CPU 기준 약 6~8분)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from img2asset.cameras import RingConfig, ring_cameras
from img2asset.fit import evaluate, fit, load_targets
from img2asset.losses import LossWeights
from img2asset.model_io import load_model
from img2asset.synth import make_synthetic_targets

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "example_device.py"
GT = {"width": 82, "depth": 24, "height": 132, "corner_r": 6, "top_r": 2, "pocket_w": 56,
      "pocket_h": 70, "pocket_z": 8, "pocket_depth": 5.5, "knob_r": 5, "knob_h": 11}


@pytest.fixture(scope="module")
def targets(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_synthetic_targets(MODEL, GT, d, resolution=384, noise=0.3)
    return d


def test_cameras_ring():
    cams = ring_cameras(RingConfig())
    assert len(cams) == 8
    assert np.allclose(cams[0].position_dir, [0, -1, 0])  # 정면 = -Y
    assert np.allclose(cams[2].position_dir, [1, 0, 0], atol=1e-9)  # 90° = +X
    for c in cams:
        assert np.isclose(np.dot(c.right, c.up), 0) and np.isclose(np.dot(c.forward, c.up), 0)


def test_flat_model_is_penalized(targets):
    """오목부를 지운 모델이 정답 모델보다 손실이 커야 한다 (평평함 퇴화 방지)."""
    model = load_model(MODEL)
    cfg, res, w = RingConfig(), 160, LossWeights()
    tg = load_targets(targets, cfg, res)
    good = evaluate(model, GT, tg, cfg, res, w).total
    flat = evaluate(model, GT | {"pocket_depth": 0.5}, tg, cfg, res, w).total
    assert flat > good * 2


def test_fit_recovers_dimensions(targets, tmp_path):
    model = load_model(MODEL)
    init = {k: v[0] for k, v in model.PARAMS.items()}
    summary = fit(MODEL, targets, tmp_path, resolution=96, maxfev=500, fixed={"height": 132.0},
                  init_params=init, log=lambda *_: None)
    p = summary["params"]
    # 외곽 치수는 소프트 IoU로 서브픽셀 정확도가 나온다.
    for k in ("width", "depth"):
        assert abs(p[k] - GT[k]) < 3.0, (k, p[k], GT[k])
    # 실루엣 안쪽 특징(홈 경계)은 깊이 잔차로만 보이므로 캔버스 픽셀 1~2개 수준이 한계다.
    canvas_px_mm = (GT["height"] + GT["knob_h"]) / (RingConfig().frame_frac * 96)
    for k in ("pocket_w", "pocket_h"):
        assert abs(p[k] - GT[k]) < 3 * canvas_px_mm, (k, p[k], GT[k], canvas_px_mm)
    assert p["pocket_depth"] > 4.0  # 초깃값 3.0에서 정답 5.5 쪽으로 이동 (오목부 복원)
    for name in ("params.json", "checkpoint.md", "report.png", "model.step", "model.stl"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "params.json").read_text())["fixed"] == {"height": 132.0}
