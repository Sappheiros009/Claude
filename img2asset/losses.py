"""피팅 오차: 실루엣 IoU + 상대 깊이(Huber) + 노멀 각도.

깊이 정렬 방향이 중요하다. 단안 깊이 추정은 스케일과 오프셋을 모르므로 affine 정렬이 필요한데,
"추정값을 렌더에 맞추면" 렌더가 평평할 때 정렬 계수가 0으로 가서 오차가 0이 되는 퇴화가 생긴다.
(모델이 오목함을 지워서 오차를 줄이는 방향으로 최적화됨)
그래서 렌더 높이를 표준화된 추정값에 맞추고(alpha * h + beta ≈ t_hat),
오차를 추정값 단위(표준편차 1)로 잰다. 평평한 모델은 잔차 분산이 1이 되어 최대 벌점을 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class LossWeights:
    iou: float = 1.0
    depth: float = 1.0
    normal: float = 0.5
    huber_delta: float = 0.1


@dataclass
class ViewLoss:
    total: float
    iou: float
    depth: float
    normal_deg: float
    alpha: float
    beta: float
    depth_residual: np.ndarray  # (R,R), t_hat 단위, 교집합 밖은 0
    normal_err_deg: np.ndarray  # (R,R), 유효 영역 밖은 0
    only_target: np.ndarray  # 모델에 없는 영역 (모델이 부족)
    only_model: np.ndarray  # 생성 이미지에 없는 영역 (모델이 넘침)


def standardize(t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = t[mask]
    if v.size < 2:
        return np.zeros_like(t)
    std = v.std()
    out = (t - v.mean()) / (std if std > 1e-12 else 1.0)
    return np.where(mask, out, 0.0)


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def robust_affine(h: np.ndarray, t: np.ndarray, delta: float, iters: int = 6) -> tuple[float, float]:
    """alpha*h + beta ≈ t 를 Huber 가중 반복 최소제곱(IRLS)으로 푼다.

    일반 최소제곱은 오목부·반사 같은 이상치가 오프셋을 끌어당겨, 정상 영역까지 잔차가 생긴다.
    IRLS는 다수의 정상 영역을 잔차 0에 맞추므로 구조 누락이 그 부위에만 나타난다.
    """
    A = np.stack([h, np.ones_like(h)], axis=1)
    wts = np.ones_like(h)
    alpha = beta = 0.0
    for _ in range(iters):
        sw = np.sqrt(wts)
        (alpha, beta), *_ = np.linalg.lstsq(A * sw[:, None], t * sw, rcond=None)
        a = np.abs(alpha * h + beta - t)
        wts = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
    return float(alpha), float(beta)


def normals_from_height(h: np.ndarray, mask: np.ndarray, pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """높이장(월드 단위)에서 카메라 좌표계 노멀을 계산한다. 3x3 이웃이 모두 물체 안인 픽셀만 유효."""
    gy, gx = np.gradient(h, pixel_size)
    n = np.stack([-gx, gy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    valid = ndimage.binary_erosion(mask, iterations=1)
    return n, valid


def view_loss(
    render_cov: np.ndarray,
    render_h: np.ndarray,
    pixel_size: float,
    target_cov: np.ndarray,
    target_that: np.ndarray,
    w: LossWeights,
) -> ViewLoss:
    """render_cov, target_cov: 0~1 커버리지. IoU는 커버리지로 계산하는 소프트 IoU다
    (경계가 픽셀 단위로 계단처럼 변하지 않아 최적화 지형이 매끄러워진다)."""
    render_mask = render_cov >= 0.5
    target_mask = target_cov >= 0.5
    inter = render_mask & target_mask
    iou = np.minimum(render_cov, target_cov).sum() / max(1e-9, np.maximum(render_cov, target_cov).sum())
    R = render_mask.shape[0]
    zeros = np.zeros((R, R))
    if inter.sum() < 16:
        return ViewLoss(
            w.iou * (1 - iou) + w.depth * 1.0 + w.normal * 1.0,
            float(iou), 1.0, 90.0, 0.0, 0.0, zeros, zeros,
            target_mask & ~render_mask, render_mask & ~target_mask,
        )

    h = render_h[inter]
    t = target_that[inter]
    alpha, beta = robust_affine(h, t, w.huber_delta)
    r = alpha * h + beta - t
    depth_l = float(huber(r, w.huber_delta).mean())
    depth_res = zeros.copy()
    depth_res[inter] = r

    # 노멀: 추정값을 렌더 월드 단위로 되돌려(높이 = (t_hat - beta) / alpha) 같은 픽셀 크기로 비교
    if alpha > 1e-9:
        t_world = (target_that - beta) / alpha
        n_r, v_r = normals_from_height(render_h, render_mask, pixel_size)
        n_t, v_t = normals_from_height(t_world, target_mask, pixel_size)
        valid = v_r & v_t
        cos = np.clip((n_r * n_t).sum(-1), -1.0, 1.0)
        normal_l = float((1.0 - cos[valid]).mean()) if valid.any() else 1.0
        ang = np.degrees(np.arccos(cos))
        nerr = np.where(valid, ang, 0.0)
        ndeg = float(ang[valid].mean()) if valid.any() else 90.0
    else:
        # 깊이 방향이 뒤집혔거나 모델이 평평함 → 최대 벌점
        normal_l, nerr, ndeg = 1.0, zeros, 90.0

    total = w.iou * (1 - iou) + w.depth * depth_l + w.normal * normal_l
    return ViewLoss(
        float(total), float(iou), depth_l, ndeg, float(alpha), float(beta),
        depth_res, nerr, target_mask & ~render_mask, render_mask & ~target_mask,
    )
