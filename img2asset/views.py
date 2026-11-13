"""시점 이미지 정규화, 턴어라운드 시트 분할, 배경 마스크 추출."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def normalize_view(
    mask: np.ndarray,
    value: np.ndarray | None,
    resolution: int,
    frame_frac: float,
    frame: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """물체 마스크의 bbox 높이가 캔버스의 frame_frac이 되도록 크기를 맞추고 가운데에 놓는다.

    렌더 결과와 생성 이미지에 똑같이 적용해서 두 쪽의 프레이밍 차이를 없앤다.
    축소할 때는 출력 픽셀마다 k x k 부표본을 최근접으로 뽑아 평균한다(box filter).
    - 마스크는 0~1 커버리지(물체가 덮은 면적 비율)가 된다 → 경계 앨리어싱이 사라져 IoU가 매끄러워짐
    - 값(깊이)은 물체에 속한 부표본만 평균한다 → 경계에서 배경과 섞이지 않음
    frame: (행 중심, 열 중심, 높이 px)를 알면 직접 넘긴다(렌더 쪽). 없으면 마스크 bbox로 잰다.
    반환: (커버리지, 값, scale). scale은 "출력 픽셀 / 입력 픽셀" 비율이다.
    """
    cov = np.zeros((resolution, resolution), np.float64)
    out_val = None if value is None else np.zeros((resolution, resolution), np.float64)
    if frame is None:
        bb = bbox(mask)
        if bb is None:
            return cov, out_val, 1.0
        y0, y1, x0, x1 = bb
        frame = (0.5 * (y0 + y1), 0.5 * (x0 + x1), float(y1 - y0 + 1))
    cy, cx, height_px = frame
    scale = frame_frac * resolution / height_px
    k = int(np.clip(np.ceil(1.0 / scale), 1, 4))
    n = resolution * k
    grid = (np.arange(n) + 0.5) / k - resolution / 2.0
    src_y = np.floor(cy + grid / scale + 0.5).astype(int)
    src_x = np.floor(cx + grid / scale + 0.5).astype(int)
    H, W = mask.shape
    vy = (src_y >= 0) & (src_y < H)
    vx = (src_x >= 0) & (src_x < W)
    yy, xx = np.meshgrid(np.clip(src_y, 0, H - 1), np.clip(src_x, 0, W - 1), indexing="ij")
    sub = (mask[yy, xx] & vy[:, None] & vx[None, :]).astype(np.float64)
    cov = sub.reshape(resolution, k, resolution, k).mean(axis=(1, 3))
    if value is not None:
        vsum = (value[yy, xx] * sub).reshape(resolution, k, resolution, k).sum(axis=(1, 3))
        cnt = sub.reshape(resolution, k, resolution, k).sum(axis=(1, 3))
        out_val = np.where(cnt > 0, vsum / np.maximum(cnt, 1), 0.0)
    return cov, out_val, scale


def background_mask(rgb: np.ndarray, threshold: float = 18.0) -> np.ndarray:
    """단색 배경(네 모서리 색의 중앙값)과 다른 픽셀을 물체로 보고, 가장 큰 연결 성분만 남긴다."""
    from scipy import ndimage

    rgb = rgb[..., :3].astype(np.float64)
    h, w = rgb.shape[:2]
    k = max(2, min(h, w) // 40)
    corners = np.concatenate(
        [
            rgb[:k, :k].reshape(-1, 3),
            rgb[:k, -k:].reshape(-1, 3),
            rgb[-k:, :k].reshape(-1, 3),
            rgb[-k:, -k:].reshape(-1, 3),
        ]
    )
    bg = np.median(corners, axis=0)
    diff = np.linalg.norm(rgb - bg, axis=-1)
    fg = diff > threshold
    fg = ndimage.binary_opening(fg, iterations=1)
    lab, n = ndimage.label(fg)
    if n == 0:
        return fg
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(keep)


def slice_sheet(sheet_path: str | Path, rows: int, cols: int, out_dir: str | Path, threshold: float = 18.0) -> list[Path]:
    """턴어라운드 시트를 rows x cols 칸으로 나눠 view_{i}.png 와 mask_{i}.png 로 저장한다.

    칸 순서는 왼쪽 위부터 행 우선(row-major)이며, 이 순서가 view 번호(0°, 45°, ...)가 된다.
    """
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = np.asarray(Image.open(sheet_path).convert("RGB"))
    H, W = img.shape[:2]
    ch, cw = H // rows, W // cols
    paths = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        cell = img[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
        m = background_mask(cell, threshold)
        Image.fromarray(cell).save(out / f"view_{i}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f"mask_{i}.png")
        paths.append(out / f"view_{i}.png")
    return paths
